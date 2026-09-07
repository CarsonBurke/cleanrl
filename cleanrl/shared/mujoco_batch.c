/* Exact MuJoCo stepping across independent models/data. No fast-math: preserve
 * the engine's arithmetic and mj_rnePostConstraint behavior used by Gymnasium.
 * Python owns every model, data, and output buffer throughout each call.
 *
 * Why a hand-rolled pool instead of `#pragma omp parallel for`
 * -----------------------------------------------------------
 * A rollout issues one batched step per ~150us of surrounding Python work
 * (policy forward, normalization, staging), so the worker team spends most of
 * its life *between* parallel regions. libgomp's default idle policy is to
 * spin (GOMP_SPINCOUNT ~ 300k), which makes a training run cost `threads`
 * whole cores no matter how little physics it does; measured on a 16-env
 * HalfCheetah rollout at 4 threads, 1630us of process CPU per step backed only
 * 614us of real work, and the spinning workers slowed the policy thread enough
 * to cost 74us/step of wall time as well. `OMP_WAIT_POLICY=passive` fixes the
 * CPU waste but libgomp's park/wake path then costs more wall time than the
 * physics it parallelizes, and intermediate GOMP_SPINCOUNT values measured
 * worse in *both* dimensions (non-monotonic: 10k spins gave 876us wall AND
 * 1379us CPU). There is no good libgomp operating point at this granularity.
 *
 * This pool therefore does exactly what is needed and nothing else:
 * - persistent threads, so no team is created or torn down per step;
 * - the submitting thread is itself a worker, so `threads=1` needs no helpers
 *   and `threads=N` needs only N-1 of them;
 * - work is claimed with one relaxed atomic increment per environment, so an
 *   env that is mid-contact (or resetting) cannot stall a whole static chunk;
 * - idle helpers spin for a bounded budget and then park on a condvar, so the
 *   common case (next step arrives promptly) pays no syscall and a paused run
 *   costs no CPU at all. The budget is tunable because its optimum depends on
 *   how loaded the machine is; see cleanrl/shared/mujoco_env.py.
 */
#include <mujoco/mujoco.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

_Static_assert(sizeof(mjtNum) == sizeof(double), "native batch buffers require double-precision MuJoCo");

#if defined(__x86_64__) || defined(__i386__)
#define CLEANRL_PAUSE() __builtin_ia32_pause()
#elif defined(__aarch64__)
#define CLEANRL_PAUSE() __asm__ __volatile__("yield" ::: "memory")
#else
#define CLEANRL_PAUSE() ((void)0)
#endif

typedef struct {
  const mjModel **models;
  mjData **data;
  const double *actions;
  double *before;
  double *positions;
  double *velocities;
  double *observations;
  int observation_offset;
  int clip_velocity;
  int count;
  int frame_skip;
} batch_t;

typedef struct {
  batch_t batch;
  int threads;          /* total workers, including the submitting thread */
  int helpers;          /* threads - 1 */
  int spin;             /* pause iterations before parking */
  pthread_t *workers;
  pthread_mutex_t lock;
  pthread_cond_t ready; /* a new generation was published */
  pthread_cond_t done;  /* `active` reached zero */
  _Atomic long generation;
  _Atomic int cursor;   /* next environment index to claim */
  _Atomic int active;   /* workers not yet finished with this generation */
  _Atomic int stop;
  int started;          /* helpers successfully created */
} pool_t;

static void step_env(const batch_t *batch, int index) {
  const mjModel *model = batch->models[index];
  mjData *data = batch->data[index];
  const int nu = model->nu, nq = model->nq, nv = model->nv;
  batch->before[index] = data->qpos[0];
  memcpy(data->ctrl, batch->actions + (size_t)index * nu, (size_t)nu * sizeof(mjtNum));
  for (int frame = 0; frame < batch->frame_skip; ++frame) {
    mj_step(model, data);
  }
  mj_rnePostConstraint(model, data);
  memcpy(batch->positions + (size_t)index * nq, data->qpos, (size_t)nq * sizeof(mjtNum));
  memcpy(batch->velocities + (size_t)index * nv, data->qvel, (size_t)nv * sizeof(mjtNum));
  /* Assemble task observations while the state is hot, without another
   * Python/NumPy dispatch. Keep the unclipped velocity buffer for Hopper's
   * health predicate. Comparisons preserve NaNs and signed zero, unlike
   * fmin/fmax, and no arithmetic or MuJoCo operation is reordered. */
  const int width = nq - batch->observation_offset;
  double *observation = batch->observations + (size_t)index * (width + nv);
  memcpy(observation, data->qpos + batch->observation_offset, (size_t)width * sizeof(mjtNum));
  if (batch->clip_velocity) {
    for (int j = 0; j < nv; ++j) {
      const double velocity = data->qvel[j];
      observation[width + j] = velocity < -10.0 ? -10.0 : velocity > 10.0 ? 10.0 : velocity;
    }
  } else {
    memcpy(observation + width, data->qvel, (size_t)nv * sizeof(mjtNum));
  }
}

/* Claim environments until the batch is exhausted. Every worker of a
 * generation runs this, including late wakers (which simply find nothing). */
static void drain(pool_t *pool) {
  const batch_t *batch = &pool->batch;
  const int count = batch->count;
  for (;;) {
    int index = atomic_fetch_add_explicit(&pool->cursor, 1, memory_order_relaxed);
    if (index >= count) {
      return;
    }
    step_env(batch, index);
  }
}

static void finish(pool_t *pool) {
  /* Release semantics: the submitter must observe our buffer writes. */
  if (atomic_fetch_sub_explicit(&pool->active, 1, memory_order_release) == 1) {
    pthread_mutex_lock(&pool->lock);
    pthread_cond_signal(&pool->done);
    pthread_mutex_unlock(&pool->lock);
  }
}

/* Block until a generation newer than `seen` is published, or the pool stops.
 * Returns the new generation, or -1 to exit. */
static long await_generation(pool_t *pool, long seen) {
  for (int spins = pool->spin;; --spins) {
    long generation = atomic_load_explicit(&pool->generation, memory_order_acquire);
    if (generation != seen) {
      return generation;
    }
    if (atomic_load_explicit(&pool->stop, memory_order_acquire)) {
      return -1;
    }
    if (spins > 0) {
      CLEANRL_PAUSE();
      continue;
    }
    /* Budget exhausted: park. Re-checking the generation under the lock is
     * what makes the wakeup lossless -- the submitter bumps the generation
     * while holding the same lock before broadcasting. */
    pthread_mutex_lock(&pool->lock);
    for (;;) {
      generation = atomic_load_explicit(&pool->generation, memory_order_acquire);
      if (generation != seen || atomic_load_explicit(&pool->stop, memory_order_acquire)) {
        break;
      }
      pthread_cond_wait(&pool->ready, &pool->lock);
    }
    pthread_mutex_unlock(&pool->lock);
    return atomic_load_explicit(&pool->stop, memory_order_acquire) ? -1 : generation;
  }
}

static void *worker(void *argument) {
  pool_t *pool = (pool_t *)argument;
  long seen = 0;
  for (;;) {
    long generation = await_generation(pool, seen);
    if (generation < 0) {
      return NULL;
    }
    seen = generation;
    drain(pool);
    finish(pool);
  }
}

/* Wait for helpers that have not finished this generation. Spins first for the
 * same reason the helpers do: at 16 envs the stragglers are microseconds away. */
static void await_completion(pool_t *pool) {
  for (int spins = pool->spin;; --spins) {
    if (atomic_load_explicit(&pool->active, memory_order_acquire) == 0) {
      return;
    }
    if (spins > 0) {
      CLEANRL_PAUSE();
      continue;
    }
    pthread_mutex_lock(&pool->lock);
    while (atomic_load_explicit(&pool->active, memory_order_acquire) != 0) {
      pthread_cond_wait(&pool->done, &pool->lock);
    }
    pthread_mutex_unlock(&pool->lock);
    return;
  }
}

void cleanrl_pool_step(void *handle) {
  pool_t *pool = (pool_t *)handle;
  if (pool->helpers == 0) {
    /* Single-threaded: no atomics, no barrier, no lock. */
    const batch_t *batch = &pool->batch;
    for (int index = 0; index < batch->count; ++index) {
      step_env(batch, index);
    }
    return;
  }
  atomic_store_explicit(&pool->cursor, 0, memory_order_relaxed);
  atomic_store_explicit(&pool->active, pool->threads, memory_order_relaxed);
  pthread_mutex_lock(&pool->lock);
  atomic_fetch_add_explicit(&pool->generation, 1, memory_order_release);
  pthread_cond_broadcast(&pool->ready);
  pthread_mutex_unlock(&pool->lock);
  drain(pool);
  /* The submitter is a worker too; if it is the last one out there is nothing
   * to wait for and no lock is taken at all. */
  if (atomic_fetch_sub_explicit(&pool->active, 1, memory_order_release) != 1) {
    await_completion(pool);
  }
}

void cleanrl_pool_destroy(void *handle) {
  pool_t *pool = (pool_t *)handle;
  if (pool == NULL) {
    return;
  }
  if (pool->started > 0) {
    pthread_mutex_lock(&pool->lock);
    atomic_store_explicit(&pool->stop, 1, memory_order_release);
    pthread_cond_broadcast(&pool->ready);
    pthread_mutex_unlock(&pool->lock);
    for (int i = 0; i < pool->started; ++i) {
      pthread_join(pool->workers[i], NULL);
    }
  }
  pthread_cond_destroy(&pool->done);
  pthread_cond_destroy(&pool->ready);
  pthread_mutex_destroy(&pool->lock);
  free(pool->workers);
  free((void *)pool->batch.models);
  free(pool->batch.data);
  free(pool);
}

/* `models`/`data` are copied; all double buffers are borrowed and must outlive
 * the pool (mujoco_env.py allocates them once). */
void *cleanrl_pool_create(int count, const mjModel **models, mjData **data,
                          const double *actions, int frame_skip, int threads, int spin,
                          double *before, double *positions, double *velocities,
                          double *observations, int observation_offset, int clip_velocity) {
  if (count <= 0 || threads <= 0 || frame_skip <= 0) {
    return NULL;
  }
  pool_t *pool = (pool_t *)calloc(1, sizeof(pool_t));
  if (pool == NULL) {
    return NULL;
  }
  pool->batch.models = (const mjModel **)malloc((size_t)count * sizeof(*models));
  pool->batch.data = (mjData **)malloc((size_t)count * sizeof(*data));
  if (pool->batch.models == NULL || pool->batch.data == NULL) {
    cleanrl_pool_destroy(pool);
    return NULL;
  }
  memcpy((void *)pool->batch.models, models, (size_t)count * sizeof(*models));
  memcpy(pool->batch.data, data, (size_t)count * sizeof(*data));
  pool->batch.actions = actions;
  pool->batch.before = before;
  pool->batch.positions = positions;
  pool->batch.velocities = velocities;
  pool->batch.observations = observations;
  pool->batch.observation_offset = observation_offset;
  pool->batch.clip_velocity = clip_velocity;
  pool->batch.count = count;
  pool->batch.frame_skip = frame_skip;
  pool->threads = threads > count ? count : threads;
  pool->helpers = pool->threads - 1;
  pool->spin = spin < 0 ? 0 : spin;
  atomic_store(&pool->generation, 0);
  if (pthread_mutex_init(&pool->lock, NULL) != 0) {
    free((void *)pool->batch.models);
    free(pool->batch.data);
    free(pool);
    return NULL;
  }
  /* Safe even if the second init never runs: PTHREAD_COND_INITIALIZER is all
   * zeros, so calloc already left `done` validly initialized for destroy. */
  if (pthread_cond_init(&pool->ready, NULL) != 0 || pthread_cond_init(&pool->done, NULL) != 0) {
    cleanrl_pool_destroy(pool);
    return NULL;
  }
  if (pool->helpers > 0) {
    pool->workers = (pthread_t *)malloc((size_t)pool->helpers * sizeof(pthread_t));
    if (pool->workers == NULL) {
      cleanrl_pool_destroy(pool);
      return NULL;
    }
    for (int i = 0; i < pool->helpers; ++i) {
      if (pthread_create(&pool->workers[i], NULL, worker, pool) != 0) {
        cleanrl_pool_destroy(pool);
        return NULL;
      }
      pool->started = i + 1;
    }
  }
  return pool;
}
