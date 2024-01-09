#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_LASX_CVT_H
#define _NPY_SIMD_LASX_CVT_H

// convert mask types to integer types
#define npyv_cvt_u8_b8(BL)   BL
#define npyv_cvt_s8_b8(BL)   BL
#define npyv_cvt_u16_b16(BL) BL
#define npyv_cvt_s16_b16(BL) BL
#define npyv_cvt_u32_b32(BL) BL
#define npyv_cvt_s32_b32(BL) BL
#define npyv_cvt_u64_b64(BL) BL
#define npyv_cvt_s64_b64(BL) BL
#define npyv_cvt_f32_b32(BL) (__m256)(BL)
#define npyv_cvt_f64_b64(BL) (__m256d)(BL)

// convert integer types to mask types
#define npyv_cvt_b8_u8(A)   A
#define npyv_cvt_b8_s8(A)   A
#define npyv_cvt_b16_u16(A) A
#define npyv_cvt_b16_s16(A) A
#define npyv_cvt_b32_u32(A) A
#define npyv_cvt_b32_s32(A) A
#define npyv_cvt_b64_u64(A) A
#define npyv_cvt_b64_s64(A) A
#define npyv_cvt_b32_f32(A) (__m256i)(A)
#define npyv_cvt_b64_f64(A) (__m256i)(A)

// convert boolean vector to integer bitfield
NPY_FINLINE npy_uint64 npyv_tobits_b8(npyv_b8 a)
{
    __m256i vmsk = __lasx_xvmskltz_b(a);
	int msk      = __lasx_xvpickve2gr_w(vmsk, 0);
	    msk     |= (__lasx_xvpickve2gr_w(vmsk, 4) << 16);
	return (npy_uint32)msk;
}
NPY_FINLINE npy_uint64 npyv_tobits_b16(npyv_b16 a)
{
   __m256i vmsk = __lasx_xvmskltz_h(a);
   int msk      = __lasx_xvpickve2gr_w(vmsk, 0);
       msk     |= (__lasx_xvpickve2gr_w(vmsk, 4) << 8);
   return msk;
}
NPY_FINLINE npy_uint64 npyv_tobits_b32(npyv_b32 a)
{
   __m256i vmsk = __lasx_xvmskltz_w(a);
   int msk      = __lasx_xvpickve2gr_w(vmsk, 0);
       msk     |= (__lasx_xvpickve2gr_w(vmsk, 4) << 4);
   return msk;
}

NPY_FINLINE npy_uint64 npyv_tobits_b64(npyv_b64 a)
{
   __m256i vmsk = __lasx_xvmskltz_d(a);
   int msk      = __lasx_xvpickve2gr_w(vmsk, 0);
       msk     |= (__lasx_xvpickve2gr_w(vmsk, 4) << 2);
   return msk;
}

// expand
NPY_FINLINE npyv_u16x2 npyv_expand_u16_u8(npyv_u8 data) {
    npyv_u16x2 r;
	__m256i tmp = (__m256i)__lasx_xvpermi_d((__m256i)data, 0x4e);
	r.val[0]    = __lasx_vext2xv_hu_bu(data);
	r.val[1]    = __lasx_vext2xv_hu_bu(tmp);
    return r;
}

NPY_FINLINE npyv_u32x2 npyv_expand_u32_u16(npyv_u16 data) {
    npyv_u32x2 r;
	__m256i tmp = (__m256i)__lasx_xvpermi_d((__m256i)data, 0x4e);
	r.val[0]    = __lasx_vext2xv_wu_hu(data);
	r.val[1]    = __lasx_vext2xv_wu_hu(tmp);
    return r;
}

// pack two 16-bit boolean into one 8-bit boolean vector
NPY_FINLINE npyv_b8 npyv_pack_b8_b16(npyv_b16 a, npyv_b16 b) {
    __m256i ab  = __lasx_xvssrarni_bu_h(b, a, 0);
    return __lasx_xvpermi_d(ab, 0xd8);
}

// pack four 32-bit boolean vectors into one 8-bit boolean vector
NPY_FINLINE npyv_b8
npyv_pack_b8_b32(npyv_b32 a, npyv_b32 b, npyv_b32 c, npyv_b32 d) {
    __m256i ab   = __lasx_xvssrarni_hu_w(b, a, 0);
    __m256i cd   = __lasx_xvssrarni_hu_w(d, c, 0);
    __m256i abcd = npyv_pack_b8_b16(ab, cd);
    return __lasx_xvpermi_d(abcd, 0xd8);
}

// pack eight 64-bit boolean vectors into one 8-bit boolean vector
NPY_FINLINE npyv_b8
npyv_pack_b8_b64(npyv_b64 a, npyv_b64 b, npyv_b64 c, npyv_b64 d,
                 npyv_b64 e, npyv_b64 f, npyv_b64 g, npyv_b64 h) {
    __m256i ab     = __lasx_xvssrarni_hu_w(b, a, 0);
    __m256i cd     = __lasx_xvssrarni_hu_w(d, c, 0);
    __m256i ef     = __lasx_xvssrarni_hu_w(f, e, 0);
    __m256i gh     = __lasx_xvssrarni_hu_w(h, g, 0);
    __m256i abcd   = __lasx_xvssrarni_hu_w(cd, ab, 0);
    __m256i efgh   = __lasx_xvssrarni_hu_w(gh, ef, 0);
	__m256i all    = npyv_pack_b8_b16(abcd, efgh);
	__m256i rev128 = __lasx_xvshuf4i_d(all, all, 0x9);
    return __lasx_xvilvl_h(rev128, all);
}

// round to nearest integer (assuming even)
#define npyv_round_s32_f32 __lasx_xvftintrne_w_s
NPY_FINLINE npyv_s32 npyv_round_s32_f64(npyv_f64 a, npyv_f64 b)
{
    return __lasx_xvftintrne_w_d(b, a);
}
#endif // _NPY_SIMD_LASX_CVT_H
