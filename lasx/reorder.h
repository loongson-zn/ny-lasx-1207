#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_LASX_REORDER_H
#define _NPY_SIMD_LASX_REORDER_H

// combine lower part of two vectors
#define npyv_combinel_u8(A, B) __lasx_xvpermi_q(B, A, 0x20)
#define npyv_combinel_s8  npyv_combinel_u8
#define npyv_combinel_u16 npyv_combinel_u8
#define npyv_combinel_s16 npyv_combinel_u8
#define npyv_combinel_u32 npyv_combinel_u8
#define npyv_combinel_s32 npyv_combinel_u8
#define npyv_combinel_u64 npyv_combinel_u8
#define npyv_combinel_s64 npyv_combinel_u8
#define npyv_combinel_f32(A, B) (__m256)__lasx_xvpermi_q(*((__m256i *)&B), *((__m256i *)&A), 0x20)
#define npyv_combinel_f64(A, B) (__m256d)__lasx_xvpermi_q(*((__m256i *)&B), *((__m256i *)&A), 0x20)

// combine higher part of two vectors
#define npyv_combineh_u8(A, B) __lasx_xvpermi_q(B, A, 0x31)
#define npyv_combineh_s8  npyv_combineh_u8
#define npyv_combineh_u16 npyv_combineh_u8
#define npyv_combineh_s16 npyv_combineh_u8
#define npyv_combineh_u32 npyv_combineh_u8
#define npyv_combineh_s32 npyv_combineh_u8
#define npyv_combineh_u64 npyv_combineh_u8
#define npyv_combineh_s64 npyv_combineh_u8
#define npyv_combineh_f32(A, B) (__m256)__lasx_xvpermi_q(*((__m256i *)&B), *((__m256i *)&A), 0x31)
#define npyv_combineh_f64(A, B) (__m256d)__lasx_xvpermi_q(*((__m256i *)&B), *((__m256i *)&A), 0x31)

// combine two vectors from lower and higher parts of two other vectors
NPY_FINLINE npyv_m256ix2 npyv__combine(__m256i a, __m256i b)
{
    npyv_m256ix2 r;
    r.val[0] = npyv_combinel_u8(a, b);  //b0a0
    r.val[1] = npyv_combineh_u8(a, b);  //b1a1
    return r;
}
NPY_FINLINE npyv_f32x2 npyv_combine_f32(__m256 a, __m256 b)
{
    npyv_f32x2 r;
    r.val[0] = npyv_combinel_f32(a, b);
    r.val[1] = npyv_combineh_f32(a, b);
    return r;
}
NPY_FINLINE npyv_f64x2 npyv_combine_f64(__m256d a, __m256d b)
{
    npyv_f64x2 r;
    r.val[0] = npyv_combinel_f64(a, b);
    r.val[1] = npyv_combineh_f64(a, b);
    return r;
}
#define npyv_combine_u8  npyv__combine
#define npyv_combine_s8  npyv__combine
#define npyv_combine_u16 npyv__combine
#define npyv_combine_s16 npyv__combine
#define npyv_combine_u32 npyv__combine
#define npyv_combine_s32 npyv__combine
#define npyv_combine_u64 npyv__combine
#define npyv_combine_s64 npyv__combine

// interleave two vectors
#define NPYV_IMPL_LASX_ZIP(T_VEC, SFX, INTR_SFX)              \
    NPY_FINLINE T_VEC##x2 npyv_zip_##SFX(T_VEC a, T_VEC b)    \
    {                                                         \
        T_VEC##x2 r;                                          \
        r.val[0] = __lasx_xvilvl_##INTR_SFX(b, a);            \
        r.val[1] = __lasx_xvilvh_##INTR_SFX(b, a);            \
        return npyv__combine(r.val[0], r.val[1]);             \
    }

NPYV_IMPL_LASX_ZIP(npyv_u8,  u8,  b)
NPYV_IMPL_LASX_ZIP(npyv_s8,  s8,  b)
NPYV_IMPL_LASX_ZIP(npyv_u16, u16, h)
NPYV_IMPL_LASX_ZIP(npyv_s16, s16, h)
NPYV_IMPL_LASX_ZIP(npyv_u32, u32, w)
NPYV_IMPL_LASX_ZIP(npyv_s32, s32, w)
NPYV_IMPL_LASX_ZIP(npyv_u64, u64, d)
NPYV_IMPL_LASX_ZIP(npyv_s64, s64, d)

NPY_FINLINE npyv_f32x2 npyv_zip_f32(__m256 a, __m256 b)
{
    npyv_f32x2 r;
    r.val[0] = (__m256)(__lasx_xvilvl_w((__m256i)b, (__m256i)a));
    r.val[1] = (__m256)(__lasx_xvilvh_w((__m256i)b, (__m256i)a));
    return npyv_combine_f32(r.val[0], r.val[1]);
}
NPY_FINLINE npyv_f64x2 npyv_zip_f64(__m256d a, __m256d b)
{
    npyv_f64x2 r;
    r.val[0] = (__m256d)(__lasx_xvilvl_d((__m256i)b, (__m256i)a));
    r.val[1] = (__m256d)(__lasx_xvilvh_d((__m256i)b, (__m256i)a));
    return npyv_combine_f64(r.val[0], r.val[1]);
}

// deinterleave two vectors
#define NPYV_IMPL_LASX_UNZIP(T_VEC, SFX, INTR_SFX)             \
    NPY_FINLINE T_VEC##x2 npyv_unzip_##SFX(T_VEC a, T_VEC b)  \
    {                                                         \
        T_VEC##x2 r;                                          \
        r.val[0] = __lasx_xvpickev_##INTR_SFX(b, a);          \
        r.val[1] = __lasx_xvpickod_##INTR_SFX(b, a);          \
        return r;                                             \
    }

NPYV_IMPL_LASX_UNZIP(npyv_u8,  u8,  b)
NPYV_IMPL_LASX_UNZIP(npyv_s8,  s8,  b)
NPYV_IMPL_LASX_UNZIP(npyv_u16, u16, h)
NPYV_IMPL_LASX_UNZIP(npyv_s16, s16, h)
NPYV_IMPL_LASX_UNZIP(npyv_u32, u32, w)
NPYV_IMPL_LASX_UNZIP(npyv_s32, s32, w)
NPYV_IMPL_LASX_UNZIP(npyv_u64, u64, d)
NPYV_IMPL_LASX_UNZIP(npyv_s64, s64, d)

NPY_FINLINE npyv_f32x2 npyv_unzip_f32(__m256 a, __m256 b)
{
    npyv_f32x2 r;
    r.val[0] = (__m256)(__lasx_xvpickev_w((__m256i)b, (__m256i)a));
    r.val[1] = (__m256)(__lasx_xvpickod_w((__m256i)b, (__m256i)a));
    return r;
}
NPY_FINLINE npyv_f64x2 npyv_unzip_f64(__m256d a, __m256d b)
{
    npyv_f64x2 r;
    r.val[0] = (__m256d)(__lasx_xvpickev_d((__m256i)b, (__m256i)a));
    r.val[1] = (__m256d)(__lasx_xvpickod_d((__m256i)b, (__m256i)a));
    return r;
}

// Reverse elements of each 64-bit lane
NPY_FINLINE npyv_u8 npyv_rev64_u8(npyv_u8 a)
{
    v32u8 idx = {                                                           
        7, 6, 5, 4, 3, 2, 1, 0,/*64*/15, 14, 13, 12, 11, 10, 9, 8,
        7, 6, 5, 4, 3, 2, 1, 0,/*64*/15, 14, 13, 12, 11, 10, 9, 8
	};
    return __lasx_xvshuf_b(a, a, (__m256i)idx);
}

#define npyv_rev64_s8 npyv_rev64_u8

NPY_FINLINE npyv_u16 npyv_rev64_u16(npyv_u16 a)
{
    v16u16 idx = {3, 2, 1, 0, 7, 6, 5, 4, 3, 2, 1, 0, 7, 6, 5, 4};
    return __lasx_xvshuf_h((__m256i)idx, a, a);
}

#define npyv_rev64_s16 npyv_rev64_u16

NPY_FINLINE npyv_u32 npyv_rev64_u32(npyv_u32 a)
{
    v8u32 idx = {1, 0, 3, 2, 1, 0, 3, 2};
    return __lasx_xvshuf_w((__m256i)idx, a, a);
}
#define npyv_rev64_s32 npyv_rev64_u32

NPY_FINLINE npyv_f32 npyv_rev64_f32(npyv_f32 a)
{
    v8u32 idx = {1, 0, 3, 2, 1, 0, 3, 2};
    return (npyv_f32)__lasx_xvshuf_w((__m256i)idx, (__m256i)a, (__m256i)a);
}

// Permuting the elements of each 128-bit lane by immediate index for
// each element.
#define npyv_permi128_u32(A, E0, E1, E2, E3)                    \
    __lasx_xvshuf4i_w(A, ((E3<<6)|(E2<<4)|(E1<<2)|(E0)))

#define npyv_permi128_s32(A, E0, E1, E2, E3) npyv_permi128_u32

#define npyv_permi128_u64(A, E0, E1)                            \
    __lasx_xvshuf4i_d(A, ((E1<<2)|(E0)))

#define npyv_permi128_s64 npyv_permi128_u64

#define npyv_permi128_f32(A, E0, E1, E2, E3)                    \
    __lasx_xvshuf4i_w(*((__m256i*)&A), ((E3<<6)|(E2<<4)|(E1<<2)|(E0)))

#define npyv_permi128_f64(A, E0, E1)                            \
    __lasx_xvshuf4i_d(*((__m256i*)&A), ((E1<<2)|(E0)))

#endif // _NPY_SIMD_LASX_REORDER_H
