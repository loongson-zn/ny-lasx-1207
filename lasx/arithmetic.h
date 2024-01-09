#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_LASX_ARITHMETIC_H
#define _NPY_SIMD_LASX_ARITHMETIC_H

/***************************
 * Addition
 ***************************/
// non-saturated
#define npyv_add_u8  __lasx_xvadd_b
#define npyv_add_s8  __lasx_xvadd_b
#define npyv_add_u16 __lasx_xvadd_h
#define npyv_add_s16 __lasx_xvadd_h
#define npyv_add_u32 __lasx_xvadd_w
#define npyv_add_s32 __lasx_xvadd_w
#define npyv_add_u64 __lasx_xvadd_d
#define npyv_add_s64 __lasx_xvadd_d
#define npyv_add_f32 __lasx_xvfadd_s
#define npyv_add_f64 __lasx_xvfadd_d

// saturated
#define npyv_adds_u8  __lasx_xvsadd_bu
#define npyv_adds_s8  __lasx_xvsadd_b
#define npyv_adds_u16 __lasx_xvsadd_hu
#define npyv_adds_s16 __lasx_xvsadd_h
// TODO: rest, after implement Packs intrins
#define npyv_adds_u32 __lasx_xvsadd_wu
#define npyv_adds_s32 __lasx_xvsadd_w
#define npyv_adds_u64 __lasx_xvsadd_du
#define npyv_adds_s64 __lasx_xvsadd_d


/***************************
 * Subtraction
 ***************************/
// non-saturated
#define npyv_sub_u8  __lasx_xvsub_b
#define npyv_sub_s8  __lasx_xvsub_b
#define npyv_sub_u16 __lasx_xvsub_h
#define npyv_sub_s16 __lasx_xvsub_h
#define npyv_sub_u32 __lasx_xvsub_w
#define npyv_sub_s32 __lasx_xvsub_w
#define npyv_sub_u64 __lasx_xvsub_d
#define npyv_sub_s64 __lasx_xvsub_d
#define npyv_sub_f32 __lasx_xvfsub_s
#define npyv_sub_f64 __lasx_xvfsub_d

// saturated
#define npyv_subs_u8  __lasx_xvssub_bu
#define npyv_subs_s8  __lasx_xvssub_b
#define npyv_subs_u16 __lasx_xvssub_hu
#define npyv_subs_s16 __lasx_xvssub_h
#define npyv_subs_u32 __lasx_xvssub_wu
#define npyv_subs_s32 __lasx_xvssub_w
#define npyv_subs_u64 __lasx_xvssub_du
#define npyv_subs_s64 __lasx_xvssub_d


/***************************
 * Multiplication
 ***************************/
// non-saturated
#define npyv_mul_u8  __lasx_xvmul_b
#define npyv_mul_s8  __lasx_xvmul_b
#define npyv_mul_u16 __lasx_xvmul_h
#define npyv_mul_s16 __lasx_xvmul_h
#define npyv_mul_u32 __lasx_xvmul_w
#define npyv_mul_s32 __lasx_xvmul_w
#define npyv_mul_f32 __lasx_xvfmul_s
#define npyv_mul_f64 __lasx_xvfmul_d


/***************************
 * Integer Division
 ***************************/
// See simd/intdiv.h for more clarification
// divide each unsigned 8-bit element by a precomputed divisor
NPY_FINLINE npyv_u8 npyv_divc_u8(npyv_u8 a, const npyv_u8x3 divisor)
{
    const __m128i bmask = __lsx_vreplgr2vr_w(0x00FF00FF);
    const __m128i shf1b = __lsx_vreplgr2vr_b(0xFFU >> __lsx_vpickve2gr_w(divisor.val[1], 0));
    const __m128i shf2b = __lsx_vreplgr2vr_b(0xFFU >> __lsx_vpickve2gr_w(divisor.val[2], 0));
    // high part of unsigned multiplication
    __m128i mulhi_even  = __lsx_vmul_h(__lsx_vand_v(a, bmask), divisor.val[0]);
    __m128i mulhi_odd   = __lsx_vmul_h(__lsx_vsrli_h(a, 8), divisor.val[0]);
            mulhi_even  = __lsx_vsrli_h(mulhi_even, 8);
    __m128i mulhi       = npyv_select_u8(bmask, mulhi_even, mulhi_odd);
    // floor(a/d)       = (mulhi + ((a-mulhi) >> sh1)) >> sh2
    __m128i q           = __lsx_vsub_b(a, mulhi);
            q           = __lsx_vand_v(__lsx_vsrl_h(q, divisor.val[1]), shf1b);
            q           = __lsx_vadd_b(mulhi, q);
            q           = __lsx_vand_v(__lsx_vsrl_h(q, divisor.val[2]), shf2b);
    return  q;
}
// divide each signed 8-bit element by a precomputed divisor (round towards zero)
NPY_FINLINE npyv_s16 npyv_divc_s16(npyv_s16 a, const npyv_s16x3 divisor);
NPY_FINLINE npyv_s8 npyv_divc_s8(npyv_s8 a, const npyv_s8x3 divisor)
{
    const __m128i bmask = __lsx_vreplgr2vr_w(0x00FF00FF);
    // instead of _mm_cvtepi8_epi16/_mm_packs_epi16 to wrap around overflow
    __m128i divc_even = npyv_divc_s16(__lsx_vsrai_h(__lsx_vslli_h(a, 8), 8), divisor);
    __m128i divc_odd  = npyv_divc_s16(__lsx_vsrai_h(a, 8), divisor);
            divc_odd  = __lsx_vslli_h(divc_odd, 8);
    return npyv_select_u8(bmask, divc_even, divc_odd);
}
// divide each unsigned 16-bit element by a precomputed divisor
NPY_FINLINE npyv_u16 npyv_divc_u16(npyv_u16 a, const npyv_u16x3 divisor)
{
    // high part of unsigned multiplication
	__m128i mulhi = __lsx_vmuh_hu(a, divisor.val[0]);
    // floor(a/d) = (mulhi + ((a-mulhi) >> sh1)) >> sh2
    __m128i q     = __lsx_vsub_h(a, mulhi);
            q     = __lsx_vsrl_h(q, divisor.val[1]);
            q     = __lsx_vadd_h(mulhi, q);
            q     = __lsx_vsrl_h(q, divisor.val[2]);
    return  q;
}
// divide each signed 16-bit element by a precomputed divisor (round towards zero)
NPY_FINLINE npyv_s16 npyv_divc_s16(npyv_s16 a, const npyv_s16x3 divisor)
{
    // high part of signed multiplication
	__m128i mulhi = __lsx_vmuh_h(a, divisor.val[0]);
    // q          = ((a + mulhi) >> sh1) - XSIGN(a)
    // trunc(a/d) = (q ^ dsign) - dsign
    __m128i q     = __lsx_vsra_h(__lsx_vadd_h(a, mulhi), divisor.val[1]);
            q     = __lsx_vsub_h(q, __lsx_vsrai_h(a, 15));
            q     = __lsx_vsub_h(__lsx_vxor_v(q, divisor.val[2]), divisor.val[2]);
    return  q;
}
// divide each unsigned 32-bit element by a precomputed divisor
NPY_FINLINE npyv_u32 npyv_divc_u32(npyv_u32 a, const npyv_u32x3 divisor)
{
    // high part of unsigned multiplication
	__m128i mulhi_even = __lsx_vsrli_d(__lsx_vmulwev_d_wu(a, divisor.val[0]), 32);
    __m128i mulhi_odd  = __lsx_vmulwev_d_wu(__lsx_vsrli_d(a, 32), divisor.val[0]);
            mulhi_odd  = __lsx_vand_v(mulhi_odd, (__m128i)(v4i32){0, -1, 0, -1});
    __m128i mulhi      = __lsx_vor_v(mulhi_even, mulhi_odd);
    // floor(a/d)      = (mulhi + ((a-mulhi) >> sh1)) >> sh2
    __m128i q          = __lsx_vsub_w(a, mulhi);
            q          = __lsx_vsrl_w(q, divisor.val[1]);
            q          = __lsx_vadd_w(mulhi, q);
            q          = __lsx_vsrl_w(q, divisor.val[2]);
    return  q;
}
// divide each signed 32-bit element by a precomputed divisor (round towards zero)
NPY_FINLINE npyv_s32 npyv_divc_s32(npyv_s32 a, const npyv_s32x3 divisor)
{
    __m128i asign      = __lsx_vsrai_w(a, 31);
    __m128i mulhi_even = __lsx_vsrli_d(__lsx_vmulwev_d_wu(a, divisor.val[0]), 32);
    __m128i mulhi_odd  = __lsx_vmulwev_d_wu(__lsx_vsrli_d(a, 32), divisor.val[0]);
            mulhi_odd  = __lsx_vand_v(mulhi_odd, (__m128i)(v4i32){0, -1, 0, -1});
    __m128i mulhi      = __lsx_vor_v(mulhi_even, mulhi_odd);
    // convert unsigned to signed high multiplication
    // mulhi - ((a < 0) ? m : 0) - ((m < 0) ? a : 0);
    const __m128i msign= __lsx_vsrai_w(divisor.val[0], 31);
    __m128i m_asign    = __lsx_vand_v(divisor.val[0], asign);
    __m128i a_msign    = __lsx_vand_v(a, msign);
            mulhi      = __lsx_vsub_w(mulhi, m_asign);
            mulhi      = __lsx_vsub_w(mulhi, a_msign);
    // q               = ((a + mulhi) >> sh1) - XSIGN(a)
    // trunc(a/d)      = (q ^ dsign) - dsign
    __m128i q          = __lsx_vsra_w(__lsx_vadd_w(a, mulhi), divisor.val[1]);
            q          = __lsx_vsub_w(q, asign);
            q          = __lsx_vsub_w(__lsx_vxor_v(q, divisor.val[2]), divisor.val[2]);
    return  q;
}
// returns the high 64 bits of unsigned 64-bit multiplication
// xref https://stackoverflow.com/a/28827013
NPY_FINLINE npyv_u64 npyv__mullhi_u64(npyv_u64 a, npyv_u64 b)
{
    __m128i lomask = npyv_setall_s64(0xffffffff);
    __m128i a_hi   = __lsx_vsrli_d(a, 32);        // a0l, a0h, a1l, a1h
    __m128i b_hi   = __lsx_vsrli_d(b, 32);        // b0l, b0h, b1l, b1h
    // compute partial products
    __m128i w0     = __lsx_vmulwev_d_wu(a, b);          // a0l*b0l, a1l*b1l
    __m128i w1     = __lsx_vmulwev_d_wu(a, b_hi);       // a0l*b0h, a1l*b1h
    __m128i w2     = __lsx_vmulwev_d_wu(a_hi, b);       // a0h*b0l, a1h*b0l
    __m128i w3     = __lsx_vmulwev_d_wu(a_hi, b_hi);    // a0h*b0h, a1h*b1h
    // sum partial products
    __m128i w0h    = __lsx_vsrli_d(w0, 32);
    __m128i s1     = __lsx_vadd_d(w1, w0h);
    __m128i s1l    = __lsx_vand_v(s1, lomask);
    __m128i s1h    = __lsx_vsrli_d(s1, 32);

    __m128i s2     = __lsx_vadd_d(w2, s1l);
    __m128i s2h    = __lsx_vsrli_d(s2, 32);

    __m128i hi     = __lsx_vadd_d(w3, s1h);
            hi     = __lsx_vadd_d(hi, s2h);
    return hi;
}
// divide each unsigned 64-bit element by a precomputed divisor
NPY_FINLINE npyv_u64 npyv_divc_u64(npyv_u64 a, const npyv_u64x3 divisor)
{
    // high part of unsigned multiplication
    __m128i mulhi = npyv__mullhi_u64(a, divisor.val[0]);
    // floor(a/d) = (mulhi + ((a-mulhi) >> sh1)) >> sh2
    __m128i q     = __lsx_vsub_d(a, mulhi);
            q     = __lsx_vsrl_d(q, divisor.val[1]);
            q     = __lsx_vadd_d(mulhi, q);
            q     = __lsx_vsrl_d(q, divisor.val[2]);
    return  q;
}
// divide each signed 64-bit element by a precomputed divisor (round towards zero)
NPY_FINLINE npyv_s64 npyv_divc_s64(npyv_s64 a, const npyv_s64x3 divisor)
{
    // high part of unsigned multiplication
    __m128i mulhi      = npyv__mullhi_u64(a, divisor.val[0]);
    // convert unsigned to signed high multiplication
    // mulhi - ((a < 0) ? m : 0) - ((m < 0) ? a : 0);
    const __m128i msign= __lsx_vslt_d(divisor.val[0], __lsx_vldi(0));
    __m128i asign      = __lsx_vslt_d(a, __lsx_vldi(0));
    __m128i m_asign    = __lsx_vand_v(divisor.val[0], asign);
    __m128i a_msign    = __lsx_vand_v(a, msign);
            mulhi      = __lsx_vsub_d(mulhi, m_asign);
            mulhi      = __lsx_vsub_d(mulhi, a_msign);
    // q               = (a + mulhi) >> sh
    __m128i q          = __lsx_vadd_d(a, mulhi);
    // emulate arithmetic right shift
    const __m128i sigb = npyv_setall_s64(1LL << 63);
            q          = __lsx_vsrl_d(__lsx_vadd_d(q, sigb), divisor.val[1]);
            q          = __lsx_vsub_d(q, __lsx_vsrl_d(sigb, divisor.val[1]));
    // q               = q - XSIGN(a)
    // trunc(a/d)      = (q ^ dsign) - dsign
            q          = __lsx_vsub_d(q, asign);
            q          = __lsx_vsub_d(__lsx_vxor_v(q, divisor.val[2]), divisor.val[2]);
    return  q;
}
/***************************
 * Division
 ***************************/
#define npyv_div_f32  __lasx_xvfdiv_s
#define npyv_div_f64  __lasx_xvfdiv_d
/***************************
 * FUSED
 ***************************/
// multiply and add, a*b + c
#define npyv_muladd_f32 __lasx_xvfmadd_s
#define npyv_muladd_f64 __lasx_xvfmadd_d
// multiply and subtract, a*b - c
#define npyv_mulsub_f32 __lasx_xvfmsub_s
#define npyv_mulsub_f64 __lasx_xvfmsub_d
// negate multiply and add, -(a*b) + c equal to -(a*b - c)
#define npyv_nmuladd_f32 __lasx_xvfnmsub_s
#define npyv_nmuladd_f64 __lasx_xvfnmsub_d
// negate multiply and subtract, -(a*b) - c equal to -(a*b +c)
#define npyv_nmulsub_f32 __lasx_xvfnmadd_s
#define npyv_nmulsub_f64 __lasx_xvfnmadd_d
 // multiply, add for odd elements and subtract even elements.
 // (a * b) -+ c
NPY_FINLINE npyv_f32 npyv_muladdsub_f32(npyv_f32 a, npyv_f32 b, npyv_f32 c)
 {
    return __lasx_xvfmadd_s(a, b, (__m256)__lasx_xvxor_v((__m256i)c, (__m256i)(v8f32){-0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0}));
 }
NPY_FINLINE npyv_f64 npyv_muladdsub_f64(npyv_f64 a, npyv_f64 b, npyv_f64 c)
 {
    return __lasx_xvfmadd_d(a, b, (__m256d)__lasx_xvxor_v((__m256i)c, (__m256i)(v4f64){-0.0, 0.0, -0.0, 0.0}));
 }

/***************************
 * Summation
 ***************************/
// reduce sum across vector
NPY_FINLINE npy_uint32 npyv_sum_u32(npyv_u32 a)
{
    __m256i t1 = __lasx_xvhaddw_du_wu(a, a);
    __m256i t2 = __lasx_xvhaddw_qu_du(t1, t1);
    return (npy_uint32)(((v8u32)t2)[0]+((v8u32)t2)[4]);
}

NPY_FINLINE npy_uint64 npyv_sum_u64(npyv_u64 a)
{
    __m256i t = __lasx_xvhaddw_qu_du(a, a);
    return (npy_uint64)(((v4u64)t)[0] + ((v4u64)t)[2]);
}

NPY_FINLINE float npyv_sum_f32(npyv_f32 a)
{
    float result = 0;
	float *fp = (float *)&a;
	for(int i = 0; i < 2; ++i){
        result += fp[i*4] + fp[i*4 + 1] + fp[i*4 + 2] + fp[i*4 +3];	
	}
    return result;
}

NPY_FINLINE double npyv_sum_f64(npyv_f64 a)
{
    double *dp = (double *)&a;
    return dp[0] + dp[1] + dp[2] + dp[3];
}

// expand the source vector and performs sum reduce
NPY_FINLINE npy_uint16 npyv_sumup_u8(npyv_u8 a)
{
    __m256i t1 = __lasx_xvhaddw_hu_bu(a, a);
    __m256i t2 = __lasx_xvhaddw_wu_hu(t1, t1);
    __m256i t3 = __lasx_xvhaddw_du_wu(t2, t2);
    __m256i t4 = __lasx_xvhaddw_qu_du(t3, t3);
    return (npy_uint16)(t4[0]+t4[1]);
}
//zn: TODO 1127
NPY_FINLINE npy_uint32 npyv_sumup_u16(npyv_u16 a)
{
    __m256i t1 = __lasx_xvhaddw_wu_hu(a, a);
    __m256i t2 = __lasx_xvhaddw_du_wu(t1, t1);
    __m256i t3 = __lasx_xvhaddw_qu_du(t2, t2);
    return (npy_uint32)(t3[0]+t3[1]);
}

#endif // _NPY_SIMD_LASX_ARITHMETIC_H
