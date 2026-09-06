/* Fused host-side forward pass for the small FP32 policy mirrors.
 *
 * cleanrl/shared/host_actor.py mirrors 64-wide policies with NumPy ufuncs. The
 * arithmetic is trivial (~1.2 MFLOP for a 3-block SiTU-sphere trunk over 16
 * rows) but it is spread over ~85 NumPy calls on (16, 64) arrays, so per-call
 * dispatch dominates. This file executes the same forward as ONE ctypes call
 * over a preassembled op graph: op codes plus integer operands in a flat int32
 * array, buffer addresses in a pointer table, both handed over once at
 * construction time. Per step the caller only re-binds the input and output
 * pointers.
 *
 * Numerics
 * --------
 * - Matmuls accumulate one sequential FP32 accumulator per output element over
 *   the reduction index (a naive dot product). No reassociation, no FMA-free
 *   requirement: -ffast-math and friends are NOT used.
 * - The only deliberate reassociation is the row sum of squares in justnorm,
 *   which uses 16 explicit partial sums (a source-level choice, so it holds
 *   with strict IEEE semantics). BLAS and torch reassociate differently, so
 *   results agree with the NumPy mirror only up to FP32 rounding.
 * - tanh/sigmoid use in-file polynomial approximations (Cephes tanhf core plus
 *   a degree-6 range-reduced expf) rather than libm scalar calls: libm's
 *   ~15-cycle scalar tanhf on the 2064 SiTU activations of a 3-block trunk
 *   would cost more than the rest of the forward. They are accurate to a few
 *   ulp, i.e. the same order as NumPy's own SIMD exp/tanh, and vectorize.
 *   exp() is only ever evaluated at non-positive arguments here, so it cannot
 *   overflow; arguments below -87 saturate, which turns sigmoid/tanh tails
 *   into 0/+-1 with absolute error below 2e-38.
 *
 * No OpenMP, no threads, no Python/NumPy C-API, no callbacks: ctypes drops the
 * GIL around the call.
 */

#include <math.h>
#include <stddef.h>
#include <stdint.h>

enum {
    CLEANRL_OP_LINEAR = 0,        /* dst = src @ w + bias                       */
    CLEANRL_OP_TANH = 1,          /* dst = tanh(src)                            */
    CLEANRL_OP_RELU = 2,          /* dst = max(src, 0)                          */
    CLEANRL_OP_LRELUSQ = 3,       /* dst = leaky_relu(src, 0.5)^2               */
    CLEANRL_OP_SITU_GLU = 4,      /* dst = situ_glu(gate, up)                   */
    CLEANRL_OP_JUSTNORM = 5,      /* dst = src / max(||src||_2, 1e-12) rowwise  */
    CLEANRL_OP_GATED_MIX = 6,     /* dst = a + g * (b - a)                      */
    CLEANRL_OP_GATED_ADD = 7,     /* dst = a + g * b                            */
    CLEANRL_OP_GATED_MIX_ACC = 8  /* dst += g * (b - a)                         */
};

/* ops[k * OP_STRIDE + 0] is the code; the remaining seven slots are buffer
 * indices and dimensions, laid out per opcode by host_graph.py. */
#define OP_STRIDE 8

typedef struct {
    const int32_t *ops;
    float **bufs;
    int32_t n_ops;
    int32_t x_slot;
    int32_t out_slot;
} cleanrl_host_graph;

/* exp(x) for x <= 0. Range reduction to |r| <= ln2/2 and a degree-6 Taylor
 * polynomial (relative error r^7/5040 < 1.2e-7 ~= 1 ulp), scaled by 2^n built
 * directly in the exponent field. The clamp keeps n >= -126 so that shift is
 * always a valid float; the comparison form (not fmaxf) is used because fmaxf
 * carries NaN-selection semantics that block vectorization without
 * -ffinite-math-only. */
static inline float exp_nonpos(float x)
{
    x = x < -87.0f ? -87.0f : x;
    const float n = rintf(x * 1.4426950408889634f);
    float r = fmaf(-n, 0.693147182464599609375f, x);
    r = fmaf(-n, -1.9046542e-9f, r);
    float p = fmaf(r, 1.0f / 720.0f, 1.0f / 120.0f);
    p = fmaf(p, r, 1.0f / 24.0f);
    p = fmaf(p, r, 1.0f / 6.0f);
    p = fmaf(p, r, 0.5f);
    p = fmaf(p, r, 1.0f);
    p = fmaf(p, r, 1.0f);
    union { uint32_t u; float f; } scale;
    scale.u = (uint32_t)((int32_t)n + 127) << 23;
    return p * scale.f;
}

/* 1 / (1 + exp(-x)), evaluated through exp(-|x|) so no argument is positive:
 * sigmoid(-|x|) = e / (1 + e) stays accurate into the exp tail. */
static inline float sigmoid_fast(float x)
{
    const float e = exp_nonpos(-fabsf(x));
    const float s = 1.0f / (1.0f + e);
    return x >= 0.0f ? s : e * s;
}

/* tanh(x): Cephes' single-precision core for |x| < 0.625 (full relative
 * accuracy near zero, which 25*tanh(u/25) needs) and (1-e)/(1+e) with
 * e = exp(-2|x|) elsewhere, where 1-e loses no significance and the tail
 * saturates to +-1 on its own. Both arms are always evaluated and blended
 * arithmetically: a plain ternary lets GCC sink the division into a guarded
 * block, which is control flow the vectorizer refuses. e lies in (0, 1], so
 * neither arm can produce a non-finite intermediate for finite x. */
static inline float tanh_fast(float x)
{
    const float z = x * x;
    float poly = fmaf(-5.70498872745e-3f, z, 2.06390887954e-2f);
    poly = fmaf(poly, z, -5.37397155531e-2f);
    poly = fmaf(poly, z, 1.33314422036e-1f);
    poly = fmaf(poly, z, -3.33332819422e-1f);
    poly = fmaf(poly * z, x, x);
    const float e = exp_nonpos(-2.0f * fabsf(x));
    const float big = copysignf((1.0f - e) / (1.0f + e), x);
    const float near = fabsf(x) < 0.625f ? 1.0f : 0.0f;
    return fmaf(near, poly - big, big);
}

/* dst(rows, out) = src(rows, in) @ w(in, out) + bias(out), all row-major.
 *
 * w arrives pre-transposed relative to nn.Linear (host_graph.py transposes
 * once per refresh) so the innermost loop walks output columns contiguously:
 * every iteration broadcasts one src element and issues independent FMAs into
 * register-resident accumulators, with no horizontal reduction and no
 * vectorization over the reduction axis (which is 17 for the input projection
 * and would need masking). The 8x16 tile keeps 8 accumulator vectors live,
 * enough to cover FMA latency, and reuses each weight vector eight times.
 * bias is never NULL: bias-free linears point at a shared zero buffer, which
 * removes a branch from the tile prologue. */
#define GEMM_TILE(RT, CT)                                                      \
    do {                                                                       \
        float acc[RT][CT];                                                     \
        for (int r = 0; r < (RT); ++r)                                         \
            for (int t = 0; t < (CT); ++t)                                     \
                acc[r][t] = bias[j + t];                                       \
        for (int p = 0; p < in; ++p) {                                         \
            const float *wp = w + (size_t)p * (size_t)out + j;                 \
            for (int r = 0; r < (RT); ++r) {                                   \
                const float xv = src[(size_t)(i + r) * (size_t)in + p];        \
                for (int t = 0; t < (CT); ++t)                                 \
                    acc[r][t] = fmaf(xv, wp[t], acc[r][t]);                    \
            }                                                                  \
        }                                                                      \
        for (int r = 0; r < (RT); ++r)                                         \
            for (int t = 0; t < (CT); ++t)                                     \
                dst[(size_t)(i + r) * (size_t)out + j + t] = acc[r][t];        \
    } while (0)

#define GEMM_COLS(RT)                                                          \
    do {                                                                       \
        int j = 0;                                                             \
        for (; j + 16 <= out; j += 16) GEMM_TILE(RT, 16);                       \
        if (out - j >= 8) { GEMM_TILE(RT, 8); j += 8; }                         \
        if (out - j >= 4) { GEMM_TILE(RT, 4); j += 4; }                         \
        for (; j < out; ++j) GEMM_TILE(RT, 1);                                  \
    } while (0)

static void gemm(float *restrict dst, const float *restrict src,
                 const float *restrict w, const float *restrict bias,
                 int rows, int in, int out)
{
    int i = 0;
    for (; i + 8 <= rows; i += 8) GEMM_COLS(8);
    if (rows - i >= 4) { GEMM_COLS(4); i += 4; }
    for (; i < rows; ++i) GEMM_COLS(1);
}

/* (4*tanh(g/4) * sigmoid(g)) * (25*tanh(u/25)), in the exact operation order
 * of host_actor.situ_glu (true divisions, not reciprocal multiplies). */
static void situ_glu(float *dst, const float *gate, const float *up, int n)
{
    for (int o = 0; o < n; ++o) {
        const float g = gate[o];
        const float capped_gate = 4.0f * tanh_fast(g / 4.0f);
        const float capped_up = 25.0f * tanh_fast(up[o] / 25.0f);
        dst[o] = (capped_gate * sigmoid_fast(g)) * capped_up;
    }
}

static void justnorm(float *dst, const float *src, int rows, int cols)
{
    for (int r = 0; r < rows; ++r) {
        const float *sr = src + (size_t)r * (size_t)cols;
        float *dr = dst + (size_t)r * (size_t)cols;
        float acc[16];
        for (int t = 0; t < 16; ++t) acc[t] = 0.0f;
        int c = 0;
        for (; c + 16 <= cols; c += 16)
            for (int t = 0; t < 16; ++t) acc[t] = fmaf(sr[c + t], sr[c + t], acc[t]);
        for (; c < cols; ++c) acc[0] = fmaf(sr[c], sr[c], acc[0]);
        float sum = 0.0f;
        for (int t = 0; t < 16; ++t) sum += acc[t];
        float norm = sqrtf(sum);
        norm = norm < 1e-12f ? 1e-12f : norm;
        for (c = 0; c < cols; ++c) dr[c] = sr[c] / norm;
    }
}

void cleanrl_host_forward(const cleanrl_host_graph *graph, const float *x, float *out)
{
    float **bufs = graph->bufs;
    /* The graph addresses the caller's input and the returned buffer through
     * the same pointer table as its own scratch; x is never written. */
    bufs[graph->x_slot] = (float *)x;
    bufs[graph->out_slot] = out;
    const int32_t *op = graph->ops;
    for (int32_t k = 0; k < graph->n_ops; ++k, op += OP_STRIDE) {
        float *dst = bufs[op[1]];
        switch (op[0]) {
        case CLEANRL_OP_LINEAR:
            gemm(dst, bufs[op[2]], bufs[op[3]], bufs[op[4]], op[5], op[6], op[7]);
            break;
        case CLEANRL_OP_TANH: {
            const float *src = bufs[op[2]];
            const int n = op[5];
            for (int o = 0; o < n; ++o) dst[o] = tanh_fast(src[o]);
            break;
        }
        case CLEANRL_OP_RELU: {
            const float *src = bufs[op[2]];
            const int n = op[5];
            for (int o = 0; o < n; ++o) dst[o] = src[o] > 0.0f ? src[o] : 0.0f;
            break;
        }
        case CLEANRL_OP_LRELUSQ: {
            const float *src = bufs[op[2]];
            const int n = op[5];
            for (int o = 0; o < n; ++o) {
                const float v = src[o];
                const float leaky = v > 0.5f * v ? v : 0.5f * v;
                dst[o] = leaky * leaky;
            }
            break;
        }
        case CLEANRL_OP_SITU_GLU:
            situ_glu(dst, bufs[op[2]], bufs[op[3]], op[5]);
            break;
        case CLEANRL_OP_JUSTNORM:
            justnorm(dst, bufs[op[2]], op[5], op[6]);
            break;
        case CLEANRL_OP_GATED_MIX: {
            const float *a = bufs[op[2]], *b = bufs[op[3]], *g = bufs[op[4]];
            const int rows = op[5], cols = op[6];
            for (int r = 0; r < rows; ++r) {
                const size_t base = (size_t)r * (size_t)cols;
                for (int c = 0; c < cols; ++c)
                    dst[base + c] = fmaf(g[c], b[base + c] - a[base + c], a[base + c]);
            }
            break;
        }
        case CLEANRL_OP_GATED_ADD: {
            const float *a = bufs[op[2]], *b = bufs[op[3]], *g = bufs[op[4]];
            const int rows = op[5], cols = op[6];
            for (int r = 0; r < rows; ++r) {
                const size_t base = (size_t)r * (size_t)cols;
                for (int c = 0; c < cols; ++c)
                    dst[base + c] = fmaf(g[c], b[base + c], a[base + c]);
            }
            break;
        }
        case CLEANRL_OP_GATED_MIX_ACC: {
            const float *a = bufs[op[2]], *b = bufs[op[3]], *g = bufs[op[4]];
            const int rows = op[5], cols = op[6];
            for (int r = 0; r < rows; ++r) {
                const size_t base = (size_t)r * (size_t)cols;
                for (int c = 0; c < cols; ++c)
                    dst[base + c] = fmaf(g[c], b[base + c] - a[base + c], dst[base + c]);
            }
            break;
        }
        default:
            return;
        }
    }
}
