/* Exact MuJoCo stepping across independent models/data. No fast-math: preserve
 * the engine's arithmetic and mj_rnePostConstraint behavior used by Gymnasium.
 * Python owns every model, data, and output buffer throughout each call. */
#include <mujoco/mujoco.h>
#include <string.h>

_Static_assert(sizeof(mjtNum) == sizeof(double), "native batch buffers require double-precision MuJoCo");

void cleanrl_mujoco_step(int count, const mjModel **models, mjData **data,
                        const double *actions, int frame_skip, int threads,
                        double *before, double *positions, double *velocities) {
  #pragma omp parallel for num_threads(threads) schedule(static) if(threads > 1)
  for (int i = 0; i < count; ++i) {
    const mjModel *m = models[i];
    mjData *d = data[i];
    before[i] = d->qpos[0];
    memcpy(d->ctrl, actions + i * m->nu, m->nu * sizeof(mjtNum));
    for (int frame = 0; frame < frame_skip; ++frame) {
      mj_step(m, d);
    }
    mj_rnePostConstraint(m, d);
    memcpy(positions + i * m->nq, d->qpos, m->nq * sizeof(mjtNum));
    memcpy(velocities + i * m->nv, d->qvel, m->nv * sizeof(mjtNum));
  }
}
