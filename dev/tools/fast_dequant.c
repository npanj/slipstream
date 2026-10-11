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

// Q5_K: 256 elements per 176 bytes
void dequant_q5_k(const uint8_t *src, float *dst, int64_t n_elements) {
    int64_t nb = n_elements / 256;
    for (int64_t i = 0; i < nb; ++i) {
        ggml_fp16_t d_raw, min_raw;
        memcpy(&d_raw, src, 2);
        memcpy(&min_raw, src + 2, 2);
        float d = fp16_to_fp32(d_raw);
        float min = fp16_to_fp32(min_raw);
        const uint8_t *scales = src + 4;
        const uint8_t *qh = src + 16;
        const uint8_t *q = src + 48;

        int is = 0;
        uint8_t sc, m;
        for (int k = 0; k < 4; ++k) {
            get_scale_min_k4(is + 0, scales, &sc, &m);
            const float d1 = d * sc; const float m1 = min * m;
            get_scale_min_k4(is + 1, scales, &sc, &m);
            const float d2 = d * sc; const float m2 = min * m;
            // qh byte l packs the high bits of qs byte l across the four
            // 64-weight blocks, LSB first: bit 2k feeds the low nibble,
            // bit 2k+1 feeds the high nibble.
            const unsigned sh1 = 2 * k, sh2 = 2 * k + 1;
            for (int l = 0; l < 32; ++l) {
                const uint32_t x1 =
                    (q[l] & 0xF) | (((qh[l] >> sh1) & 1u) << 4);
                *dst++ = d1 * x1 - m1;
            }
            for (int l = 0; l < 32; ++l) {
                const uint32_t x2 =
                    (q[l] >> 4) | (((qh[l] >> sh2) & 1u) << 4);
                *dst++ = d2 * x2 - m2;
            }
            q += 32; is += 2;
        }
        src += 176;
    }
}

// IQ4_NL: 32 elements per 18 bytes. Codes index a fixed 16-entry grid;
// the grid below was recovered empirically against libggml-base
// (dequantize_row_iq4_nl): every code's y/d ratio is exactly constant.
static const int8_t kIq4NlGrid[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10,
    1, 13, 25, 38, 53, 69, 89, 113,
};

void dequant_iq4_nl(const uint8_t *src, float *dst, int64_t n_elements) {
    int64_t nb = n_elements / 32;
    for (int64_t i = 0; i < nb; ++i) {
        ggml_fp16_t d_raw;
        memcpy(&d_raw, src, 2);
        float d = fp16_to_fp32(d_raw);
        src += 2;
        for (int l = 0; l < 16; ++l) {
            dst[i * 32 + l] = d * kIq4NlGrid[src[l] & 0xF];
            dst[i * 32 + l + 16] = d * kIq4NlGrid[src[l] >> 4];
        }
        src += 16;
    }
}

// Q3_K: 256 elements per 110 bytes. Layout: hmask[32] (high bits),
// qs[64] (low 2 bits), scales[12], d. Ported from llama.cpp ggml
// (dequantize_row_q3_K, MIT); differential-tested bit-exact against
// libggml-base.
void dequant_q3_k(const uint8_t *src, float *dst, int64_t n_elements) {
    static const uint32_t kmask1 = 0x03030303u;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    int64_t nb = n_elements / 256;
    for (int64_t i = 0; i < nb; ++i) {
        ggml_fp16_t d_raw;
        memcpy(&d_raw, src + 108, 2);
        float d_all = fp16_to_fp32(d_raw);
        const uint8_t *hm = src;
        const uint8_t *q = src + 32;
        uint32_t aux[4];
        int8_t scales[16];
        memcpy(aux, src + 96, 12);
        uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        memcpy(scales, aux, 16);

        int is = 0;
        uint8_t m = 1;
        for (int n = 0; n < 256; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                float dl = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; ++l) {
                    *dst++ = dl * ((int8_t)((q[l + 0] >> shift) & 3) -
                                   ((hm[l + 0] & m) ? 0 : 4));
                }
                dl = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; ++l) {
                    *dst++ = dl * ((int8_t)((q[l + 16] >> shift) & 3) -
                                   ((hm[l + 16] & m) ? 0 : 4));
                }
                shift += 2;
                m <<= 1;
            }
            q += 32;
        }
        src += 110;
    }
}

// IQ4_XS: 256 elements per 136 bytes. Layout: d, scales_h, scales_l[4],
// qs[128]; values index the IQ4_NL grid. Ported from llama.cpp ggml
// (dequantize_row_iq4_xs, MIT); differential-tested bit-exact.
void dequant_iq4_xs(const uint8_t *src, float *dst, int64_t n_elements) {
    int64_t nb = n_elements / 256;
    for (int64_t i = 0; i < nb; ++i) {
        ggml_fp16_t d_raw;
        uint16_t h;
        memcpy(&d_raw, src, 2);
        memcpy(&h, src + 2, 2);
        float d = fp16_to_fp32(d_raw);
        const uint8_t *scales_l = src + 4;
        const uint8_t *qs = src + 8;
        for (int ib = 0; ib < 8; ++ib) {
            int ls = ((scales_l[ib / 2] >> (4 * (ib % 2))) & 0xF) |
                     (((h >> (2 * ib)) & 3) << 4);
            float dl = d * (ls - 32);
            for (int j = 0; j < 16; ++j) {
                dst[ib * 32 + j] = dl * kIq4NlGrid[qs[ib * 16 + j] & 0xF];
                dst[ib * 32 + j + 16] =
                    dl * kIq4NlGrid[qs[ib * 16 + j] >> 4];
            }
        }
        dst += 256;
        src += 136;
    }
}

// IQ3_S grid: 512 entries of four int8 values, from llama.cpp ggml
// (ggml-common.h iq3s_grid, MIT).
static const uint32_t kIq3sGrid[512] = {
    0x01010101, 0x01010103, 0x01010105, 0x0101010b, 0x0101010f, 0x01010301, 0x01010303, 0x01010305,
    0x01010309, 0x0101030d, 0x01010501, 0x01010503, 0x0101050b, 0x01010707, 0x01010901, 0x01010905,
    0x0101090b, 0x0101090f, 0x01010b03, 0x01010b07, 0x01010d01, 0x01010d05, 0x01010f03, 0x01010f09,
    0x01010f0f, 0x01030101, 0x01030103, 0x01030105, 0x01030109, 0x01030301, 0x01030303, 0x0103030b,
    0x01030501, 0x01030507, 0x0103050f, 0x01030703, 0x0103070b, 0x01030909, 0x01030d03, 0x01030d0b,
    0x01030f05, 0x01050101, 0x01050103, 0x0105010b, 0x0105010f, 0x01050301, 0x01050307, 0x0105030d,
    0x01050503, 0x0105050b, 0x01050701, 0x01050709, 0x01050905, 0x0105090b, 0x0105090f, 0x01050b03,
    0x01050b07, 0x01050f01, 0x01050f07, 0x01070107, 0x01070303, 0x0107030b, 0x01070501, 0x01070505,
    0x01070703, 0x01070707, 0x0107070d, 0x01070909, 0x01070b01, 0x01070b05, 0x01070d0f, 0x01070f03,
    0x01070f0b, 0x01090101, 0x01090307, 0x0109030f, 0x01090503, 0x01090509, 0x01090705, 0x01090901,
    0x01090907, 0x01090b03, 0x01090f01, 0x010b0105, 0x010b0109, 0x010b0501, 0x010b0505, 0x010b050d,
    0x010b0707, 0x010b0903, 0x010b090b, 0x010b090f, 0x010b0d0d, 0x010b0f07, 0x010d010d, 0x010d0303,
    0x010d0307, 0x010d0703, 0x010d0b05, 0x010d0f03, 0x010f0101, 0x010f0105, 0x010f0109, 0x010f0501,
    0x010f0505, 0x010f050d, 0x010f0707, 0x010f0b01, 0x010f0b09, 0x03010101, 0x03010103, 0x03010105,
    0x03010109, 0x03010301, 0x03010303, 0x03010307, 0x0301030b, 0x0301030f, 0x03010501, 0x03010505,
    0x03010703, 0x03010709, 0x0301070d, 0x03010b09, 0x03010b0d, 0x03010d03, 0x03010f05, 0x03030101,
    0x03030103, 0x03030107, 0x0303010d, 0x03030301, 0x03030309, 0x03030503, 0x03030701, 0x03030707,
    0x03030903, 0x03030b01, 0x03030b05, 0x03030f01, 0x03030f0d, 0x03050101, 0x03050305, 0x0305030b,
    0x0305030f, 0x03050501, 0x03050509, 0x03050705, 0x03050901, 0x03050907, 0x03050b0b, 0x03050d01,
    0x03050f05, 0x03070103, 0x03070109, 0x0307010f, 0x03070301, 0x03070307, 0x03070503, 0x0307050f,
    0x03070701, 0x03070709, 0x03070903, 0x03070d05, 0x03070f01, 0x03090107, 0x0309010b, 0x03090305,
    0x03090309, 0x03090703, 0x03090707, 0x03090905, 0x0309090d, 0x03090b01, 0x03090b09, 0x030b0103,
    0x030b0301, 0x030b0307, 0x030b0503, 0x030b0701, 0x030b0705, 0x030b0b03, 0x030d0501, 0x030d0509,
    0x030d050f, 0x030d0909, 0x030d090d, 0x030f0103, 0x030f0107, 0x030f0301, 0x030f0305, 0x030f0503,
    0x030f070b, 0x030f0903, 0x030f0d05, 0x030f0f01, 0x05010101, 0x05010103, 0x05010107, 0x0501010b,
    0x0501010f, 0x05010301, 0x05010305, 0x05010309, 0x0501030d, 0x05010503, 0x05010507, 0x0501050f,
    0x05010701, 0x05010705, 0x05010903, 0x05010907, 0x0501090b, 0x05010b01, 0x05010b05, 0x05010d0f,
    0x05010f01, 0x05010f07, 0x05010f0b, 0x05030101, 0x05030105, 0x05030301, 0x05030307, 0x0503030f,
    0x05030505, 0x0503050b, 0x05030703, 0x05030709, 0x05030905, 0x05030b03, 0x05050103, 0x05050109,
    0x0505010f, 0x05050503, 0x05050507, 0x05050701, 0x0505070f, 0x05050903, 0x05050b07, 0x05050b0f,
    0x05050f03, 0x05050f09, 0x05070101, 0x05070105, 0x0507010b, 0x05070303, 0x05070505, 0x05070509,
    0x05070703, 0x05070707, 0x05070905, 0x05070b01, 0x05070d0d, 0x05090103, 0x0509010f, 0x05090501,
    0x05090507, 0x05090705, 0x0509070b, 0x05090903, 0x05090f05, 0x05090f0b, 0x050b0109, 0x050b0303,
    0x050b0505, 0x050b070f, 0x050b0901, 0x050b0b07, 0x050b0f01, 0x050d0101, 0x050d0105, 0x050d010f,
    0x050d0503, 0x050d0b0b, 0x050d0d03, 0x050f010b, 0x050f0303, 0x050f050d, 0x050f0701, 0x050f0907,
    0x050f0b01, 0x07010105, 0x07010303, 0x07010307, 0x0701030b, 0x0701030f, 0x07010505, 0x07010703,
    0x07010707, 0x0701070b, 0x07010905, 0x07010909, 0x0701090f, 0x07010b03, 0x07010d07, 0x07010f03,
    0x07030103, 0x07030107, 0x0703010b, 0x07030309, 0x07030503, 0x07030507, 0x07030901, 0x07030d01,
    0x07030f05, 0x07030f0d, 0x07050101, 0x07050305, 0x07050501, 0x07050705, 0x07050709, 0x07050b01,
    0x07070103, 0x07070301, 0x07070309, 0x07070503, 0x07070507, 0x0707050f, 0x07070701, 0x07070903,
    0x07070907, 0x0707090f, 0x07070b0b, 0x07070f07, 0x07090107, 0x07090303, 0x0709030d, 0x07090505,
    0x07090703, 0x07090b05, 0x07090d01, 0x07090d09, 0x070b0103, 0x070b0301, 0x070b0305, 0x070b050b,
    0x070b0705, 0x070b0909, 0x070b0b0d, 0x070b0f07, 0x070d030d, 0x070d0903, 0x070f0103, 0x070f0107,
    0x070f0501, 0x070f0505, 0x070f070b, 0x09010101, 0x09010109, 0x09010305, 0x09010501, 0x09010509,
    0x0901050f, 0x09010705, 0x09010903, 0x09010b01, 0x09010f01, 0x09030105, 0x0903010f, 0x09030303,
    0x09030307, 0x09030505, 0x09030701, 0x0903070b, 0x09030907, 0x09030b03, 0x09030b0b, 0x09050103,
    0x09050107, 0x09050301, 0x0905030b, 0x09050503, 0x09050707, 0x09050901, 0x09050b0f, 0x09050d05,
    0x09050f01, 0x09070109, 0x09070303, 0x09070307, 0x09070501, 0x09070505, 0x09070703, 0x0907070b,
    0x09090101, 0x09090105, 0x09090509, 0x0909070f, 0x09090901, 0x09090f03, 0x090b010b, 0x090b010f,
    0x090b0503, 0x090b0d05, 0x090d0307, 0x090d0709, 0x090d0d01, 0x090f0301, 0x090f030b, 0x090f0701,
    0x090f0907, 0x090f0b03, 0x0b010105, 0x0b010301, 0x0b010309, 0x0b010505, 0x0b010901, 0x0b010909,
    0x0b01090f, 0x0b010b05, 0x0b010d0d, 0x0b010f09, 0x0b030103, 0x0b030107, 0x0b03010b, 0x0b030305,
    0x0b030503, 0x0b030705, 0x0b030f05, 0x0b050101, 0x0b050303, 0x0b050507, 0x0b050701, 0x0b05070d,
    0x0b050b07, 0x0b070105, 0x0b07010f, 0x0b070301, 0x0b07050f, 0x0b070909, 0x0b070b03, 0x0b070d0b,
    0x0b070f07, 0x0b090103, 0x0b090109, 0x0b090501, 0x0b090705, 0x0b09090d, 0x0b0b0305, 0x0b0b050d,
    0x0b0b0b03, 0x0b0b0b07, 0x0b0d0905, 0x0b0f0105, 0x0b0f0109, 0x0b0f0505, 0x0d010303, 0x0d010307,
    0x0d01030b, 0x0d010703, 0x0d010707, 0x0d010d01, 0x0d030101, 0x0d030501, 0x0d03050f, 0x0d030d09,
    0x0d050305, 0x0d050709, 0x0d050905, 0x0d050b0b, 0x0d050d05, 0x0d050f01, 0x0d070101, 0x0d070309,
    0x0d070503, 0x0d070901, 0x0d09050b, 0x0d090907, 0x0d090d05, 0x0d0b0101, 0x0d0b0107, 0x0d0b0709,
    0x0d0b0d01, 0x0d0d010b, 0x0d0d0901, 0x0d0f0303, 0x0d0f0307, 0x0f010101, 0x0f010109, 0x0f01010f,
    0x0f010501, 0x0f010505, 0x0f01070d, 0x0f010901, 0x0f010b09, 0x0f010d05, 0x0f030105, 0x0f030303,
    0x0f030509, 0x0f030907, 0x0f03090b, 0x0f050103, 0x0f050109, 0x0f050301, 0x0f05030d, 0x0f050503,
    0x0f050701, 0x0f050b03, 0x0f070105, 0x0f070705, 0x0f07070b, 0x0f070b07, 0x0f090103, 0x0f09010b,
    0x0f090307, 0x0f090501, 0x0f090b01, 0x0f0b0505, 0x0f0b0905, 0x0f0d0105, 0x0f0d0703, 0x0f0f0101,
};

// IQ3_S: 256 elements per 110 bytes. Layout: d, qs[64], qh[8], signs[32],
// scales[4]. Ported from llama.cpp ggml (dequantize_row_iq3_s, MIT);
// differential-tested bit-exact against libggml-base.
void dequant_iq3_s(const uint8_t *src, float *dst, int64_t n_elements) {
    static const uint8_t kmask[8] = {1, 2, 4, 8, 16, 32, 64, 128};
    int64_t nb = n_elements / 256;
    for (int64_t i = 0; i < nb; ++i) {
        ggml_fp16_t d_raw;
        memcpy(&d_raw, src, 2);
        float d = fp16_to_fp32(d_raw);
        const uint8_t *qs = src + 2;
        const uint8_t *qh = src + 66;
        const uint8_t *signs = src + 74;
        const uint8_t *scales = src + 106;
        for (int ib32 = 0; ib32 < 8; ib32 += 2) {
            float db1 = d * (1 + 2 * (scales[ib32 / 2] & 0xF));
            float db2 = d * (1 + 2 * (scales[ib32 / 2] >> 4));
            const uint8_t *s = signs;
            for (int l = 0; l < 4; ++l) {
                const uint8_t *g1 =
                    (const uint8_t *)(kIq3sGrid + (qs[2 * l] | ((qh[0] << (8 - 2 * l)) & 256)));
                const uint8_t *g2 =
                    (const uint8_t *)(kIq3sGrid + (qs[2 * l + 1] | ((qh[0] << (7 - 2 * l)) & 256)));
                for (int j = 0; j < 4; ++j) {
                    dst[j] = db1 * g1[j] * ((s[l] & kmask[j]) ? -1.f : 1.f);
                    dst[j + 4] = db1 * g2[j] * ((s[l] & kmask[j + 4]) ? -1.f : 1.f);
                }
                dst += 8;
            }
            qs += 8;
            s = signs + 4;
            for (int l = 0; l < 4; ++l) {
                const uint8_t *g1 =
                    (const uint8_t *)(kIq3sGrid + (qs[2 * l] | ((qh[1] << (8 - 2 * l)) & 256)));
                const uint8_t *g2 =
                    (const uint8_t *)(kIq3sGrid + (qs[2 * l + 1] | ((qh[1] << (7 - 2 * l)) & 256)));
                for (int j = 0; j < 4; ++j) {
                    dst[j] = db2 * g1[j] * ((s[l] & kmask[j]) ? -1.f : 1.f);
                    dst[j + 4] = db2 * g2[j] * ((s[l] & kmask[j + 4]) ? -1.f : 1.f);
                }
                dst += 8;
            }
            qh += 2;
            qs += 8;
            signs += 8;
        }
        src += 110;
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
