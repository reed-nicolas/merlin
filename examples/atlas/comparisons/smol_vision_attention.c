#include "smol_vision_attention.h"
#include "smol_host_bits.h"

#include <fenv.h>
#include <float.h>
#include <math.h>

/* Arithmetic contract: PyTorch2.10 CPU FlashAttentionKernel, BF16 CPU16.
 * Retain512-key running-max blocks, FLOAT lane sums, BF16 unnormalized
 * exponentials, and final FLOAT normalization. This is not eager attention.
 */
static float vision_exp(float value) {
    if (value < -87.3365478515625f) return 0.0f;
    /* log2(e) requires all23 fraction bits, unlike BF16 operands. */
    uint32_t log_bits = UINT32_C(0x3fb8aa3b);
    float log2e;
    memcpy(&log2e, &log_bits, sizeof(log2e));
    float source = value * log2e;
    float fraction = source - floorf(source);
    float correction = fmaf(fraction, -0.079204240219773236f, -0.22433836478672356f);
    correction = fmaf(fraction, correction, 0.30354260500649682f);
    correction = fmaf(fraction, correction, 0.00010703434948458272f);
    source -= correction;
    float encoded = fmaf(8388608.0f, source, 1065353216.0f);
    uint32_t bits = (uint32_t)(int32_t)encoded;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static int overlap(const uint16_t *left, size_t left_words,
                   const uint16_t *right, size_t right_words) {
    uintptr_t a = (uintptr_t)left, b = (uintptr_t)right;
    return a < b + right_words * sizeof(uint16_t)
        && b < a + left_words * sizeof(uint16_t);
}

int smol_vision_attention_cpu16_bf16(
    const uint16_t *query, const uint16_t *key, const uint16_t *value,
    size_t rows, size_t keys, size_t heads, uint16_t *output) {
    if (!query || !key || !value || !output || !rows || rows > 1024
        || !keys || keys > 1024 || keys % 16 || !heads || heads > 12
        || FLT_RADIX != 2 || FLT_MANT_DIG != 24 || sizeof(float) != 4
        || fegetround() != FE_TONEAREST) return -1;
    size_t width = heads * 64, q_words = rows * width, kv_words = keys * width;
    if (overlap(output, q_words, query, q_words)
        || overlap(output, q_words, key, kv_words)
        || overlap(output, q_words, value, kv_words)) return -1;
    for (size_t i = 0; i < q_words; ++i) {
        if ((query[i] & UINT16_C(0x7f80)) == UINT16_C(0x7f80)) return -1;
    }
    for (size_t i = 0; i < kv_words; ++i) {
        if ((key[i] & UINT16_C(0x7f80)) == UINT16_C(0x7f80)
            || (value[i] & UINT16_C(0x7f80)) == UINT16_C(0x7f80)) return -1;
    }
    for (size_t head = 0; head < heads; ++head) {
        for (size_t row = 0; row < rows; ++row) {
            float maximum = -INFINITY, denominator = 0.0f, accumulator[64] = {0};
            for (size_t block = 0; block < keys; block += 512) {
                size_t count = keys - block < 512 ? keys - block : 512;
                float scores[512], lane_sums[16] = {0};
                uint16_t exponentials[512];
                float next_maximum = maximum;
                for (size_t column = 0; column < count; ++column) {
                    float dot = 0.0f;
                    size_t q_offset = row * width + head * 64;
                    size_t k_offset = (block + column) * width + head * 64;
                    /* Native BF16->FP32 MKL QK uses ascending FMA order. */
                    for (size_t d = 0; d < 64; ++d) {
                        dot = fmaf(bf16_to_float(query[q_offset + d]),
                                   bf16_to_float(key[k_offset + d]), dot);
                    }
                    scores[column] = dot * 0.125f + 0.0f;
                    if (!isfinite(scores[column])) return -1;
                    if (scores[column] > next_maximum) next_maximum = scores[column];
                }
                for (size_t column = 0; column < count; ++column) {
                    float exponential = vision_exp(scores[column] - next_maximum);
                    lane_sums[column % 16] += exponential;
                    exponentials[column] = float_to_bf16(exponential);
                }
                for (size_t stride = 8; stride; stride /= 2) {
                    for (size_t lane = 0; lane < stride; ++lane) {
                        lane_sums[lane] += lane_sums[lane + stride];
                    }
                }
                float prior_scale = expf(maximum - next_maximum);
                /* Native AVX512 contracts the running-sum update into one FMA. */
                denominator = fmaf(prior_scale, denominator, lane_sums[0]);
                maximum = next_maximum;
                for (size_t d = 0; d < 64; ++d) {
                    float current = block ? accumulator[d] * prior_scale : 0.0f;
                    /* The native float-output GEMM splits K into256 terms.
                     * Each ascending dot starts at zero, then adds beta*C.
                     */
                    for (size_t part = 0; part < count; part += 256) {
                        size_t end = count - part < 256 ? count : part + 256;
                        float sum = 0.0f;
                        for (size_t column = part; column < end; ++column) {
                            sum = fmaf(bf16_to_float(exponentials[column]),
                                       bf16_to_float(value[(block + column) * width + head * 64 + d]), sum);
                        }
                        current = block || part ? sum + current : sum;
                    }
                    accumulator[d] = current;
                }
            }
            float reciprocal = 1.0f / denominator;
            if (!isfinite(reciprocal)) return -1;
            for (size_t d = 0; d < 64; ++d) {
                if (!isfinite(accumulator[d])) return -1;
                output[row * width + head * 64 + d] = float_to_bf16(accumulator[d] * reciprocal);
            }
        }
    }
    return 0;
}
