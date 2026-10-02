#include <stdint.h>
#include <string.h>
#include <math.h>

typedef uint16_t ggml_fp16_t;

static inline float fp16_to_fp32(ggml_fp16_t h) {
    uint32_t sign = ((uint32_t)h & 0x8000) << 16;
    int32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x03FF;

    if (exp == 0) {
        if (mant == 0) {
            float f;
            uint32_t u = sign;
            memcpy(&f, &u, 4);
            return f;
        }
        // Subnormal: normalize so the implicit leading bit is set.
        exp = 1;
        while (!(mant & 0x0400)) {
            mant <<= 1;
            exp--;
        }
        mant &= 0x03FF;
    } else if (exp == 31) {
        float f;
        uint32_t u = sign | 0x7F800000 | (mant << 13);
        memcpy(&f, &u, 4);
        return f;
    }
    exp = exp + (127 - 15);
    uint32_t u = sign | ((uint32_t)exp << 23) | (mant << 13);
    float f;
    memcpy(&f, &u, 4);
    return f;
}

// Q4_0: 32 elements per 18 bytes
void dequant_q4_0(const uint8_t *src, float *dst, int64_t n_elements) {
    int64_t nb = n_elements / 32;
    for (int64_t i = 0; i < nb; ++i) {
        ggml_fp16_t d_raw;
        memcpy(&d_raw, src, 2);
        float d = fp16_to_fp32(d_raw);
        src += 2;
        for (int l = 0; l < 16; ++l) {
            dst[i * 32 + l] = ((src[l] & 0x0F) - 8.0f) * d;
            dst[i * 32 + l + 16] = ((src[l] >> 4) - 8.0f) * d;
        }
        src += 16;
    }
}

// Q4_1: 32 elements per 20 bytes
void dequant_q4_1(const uint8_t *src, float *dst, int64_t n_elements) {
    int64_t nb = n_elements / 32;
    for (int64_t i = 0; i < nb; ++i) {
        ggml_fp16_t d_raw, m_raw;
        memcpy(&d_raw, src, 2);
        memcpy(&m_raw, src + 2, 2);
        float d = fp16_to_fp32(d_raw);
        float m = fp16_to_fp32(m_raw);
        src += 4;
        for (int l = 0; l < 16; ++l) {
            dst[i * 32 + l] = (src[l] & 0x0F) * d + m;
            dst[i * 32 + l + 16] = (src[l] >> 4) * d + m;
        }
        src += 16;
    }
}

// Q8_0: 32 elements per 34 bytes
void dequant_q8_0(const uint8_t *src, float *dst, int64_t n_elements) {
    int64_t nb = n_elements / 32;
    for (int64_t i = 0; i < nb; ++i) {
        ggml_fp16_t d_raw;
        memcpy(&d_raw, src, 2);
        float d = fp16_to_fp32(d_raw);
        src += 2;
        const int8_t *q = (const int8_t *)src;
        for (int l = 0; l < 32; ++l) {
            dst[i * 32 + l] = (float)q[l] * d;
        }
        src += 32;
    }
}

static inline void get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

// Q4_K: 256 elements per 144 bytes
void dequant_q4_k(const uint8_t *src, float *dst, int64_t n_elements) {
    int64_t nb = n_elements / 256;
    for (int64_t i = 0; i < nb; ++i) {
        ggml_fp16_t d_raw, min_raw;
        memcpy(&d_raw, src, 2);
        memcpy(&min_raw, src + 2, 2);
        float d = fp16_to_fp32(d_raw);
        float min = fp16_to_fp32(min_raw);
        const uint8_t *scales = src + 4;
        const uint8_t *q = src + 16;

        int is = 0;
        uint8_t sc, m;
        for (int j = 0; j < 256; j += 64) {
            get_scale_min_k4(is + 0, scales, &sc, &m);
            const float d1 = d * sc; const float m1 = min * m;
            get_scale_min_k4(is + 1, scales, &sc, &m);
            const float d2 = d * sc; const float m2 = min * m;
            for (int l = 0; l < 32; ++l) *dst++ = d1 * (q[l] & 0xF) - m1;
            for (int l = 0; l < 32; ++l) *dst++ = d2 * (q[l] >> 4) - m2;
            q += 32; is += 2;
        }
        src += 144;
    }
}

// Q5_0: 32 elements per 22 bytes
void dequant_q5_0(const uint8_t *src, float *dst, int64_t n_elements) {
    int64_t nb = n_elements / 32;
    for (int64_t i = 0; i < nb; ++i) {
        ggml_fp16_t d_raw;
        memcpy(&d_raw, src, 2);
        float d = fp16_to_fp32(d_raw);
        uint32_t qh;
        memcpy(&qh, src + 2, 4);
        const uint8_t *qs = src + 6;

        for (int j = 0; j < 16; ++j) {
            const uint8_t xh_0 = ((qh >> (j + 0)) << 4) & 0x10;
            const uint8_t xh_1 = ((qh >> (j + 12))) & 0x10;
            const int32_t x0 = ((qs[j] & 0x0F) | xh_0) - 16;
            const int32_t x1 = ((qs[j] >> 4) | xh_1) - 16;
            dst[i * 32 + j] = x0 * d;
            dst[i * 32 + j + 16] = x1 * d;
        }
        src += 22;
    }
}

// Q6_K: 256 elements per 210 bytes
void dequant_q6_k(const uint8_t *src, float *dst, int64_t n_elements) {
    int64_t nb = n_elements / 256;
    for (int64_t i = 0; i < nb; ++i) {
        const uint8_t *ql = src;
        const uint8_t *qh = src + 128;
        const int8_t *sc = (const int8_t *)(src + 192);
        ggml_fp16_t d_raw;
        memcpy(&d_raw, src + 208, 2);
        float d = fp16_to_fp32(d_raw);

        for (int n = 0; n < 256; n += 128) {
            for (int l = 0; l < 32; ++l) {
                int is = l / 16;
                const int8_t q1 = (int8_t)((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                dst[l + 0] = d * sc[is + 0] * q1;
                dst[l + 32] = d * sc[is + 2] * q2;
                dst[l + 64] = d * sc[is + 4] * q3;
                dst[l + 96] = d * sc[is + 6] * q4;
            }
            dst += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
        src += 210;
    }
}
