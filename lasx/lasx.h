#ifndef _NPY_SIMD_H_
    #error "Not a standalone header"
#endif

#define NPY_SIMD 256
#define NPY_SIMD_WIDTH 32
#define NPY_SIMD_F64 1
#define NPY_SIMD_F32 1
#define NPY_SIMD_FMA3 1
#define NPY_SIMD_BIGENDIAN 0
#define NPY_SIMD_CMPSIGNAL 1

typedef __m256i npyv_u8;
typedef __m256i npyv_s8;
typedef __m256i npyv_u16;
typedef __m256i npyv_s16;
typedef __m256i npyv_u32;
typedef __m256i npyv_s32;
typedef __m256i npyv_u64;
typedef __m256i npyv_s64;
typedef __m256 npyv_f32;
typedef __m256d npyv_f64;

typedef __m256i npyv_b8;
typedef __m256i npyv_b16;
typedef __m256i npyv_b32;
typedef __m256i npyv_b64;

typedef struct { __m256i val[2]; } npyv_m256ix2;
typedef npyv_m256ix2 npyv_u8x2;
typedef npyv_m256ix2 npyv_s8x2;
typedef npyv_m256ix2 npyv_u16x2;
typedef npyv_m256ix2 npyv_s16x2;
typedef npyv_m256ix2 npyv_u32x2;
typedef npyv_m256ix2 npyv_s32x2;
typedef npyv_m256ix2 npyv_u64x2;
typedef npyv_m256ix2 npyv_s64x2;

typedef struct { __m256i val[3]; } npyv_m256ix3;
typedef npyv_m256ix3 npyv_u8x3;
typedef npyv_m256ix3 npyv_s8x3;
typedef npyv_m256ix3 npyv_u16x3;
typedef npyv_m256ix3 npyv_s16x3;
typedef npyv_m256ix3 npyv_u32x3;
typedef npyv_m256ix3 npyv_s32x3;
typedef npyv_m256ix3 npyv_u64x3;
typedef npyv_m256ix3 npyv_s64x3;

typedef struct { __m256  val[2]; } npyv_f32x2;
typedef struct { __m256d val[2]; } npyv_f64x2;
typedef struct { __m256  val[3]; } npyv_f32x3;
typedef struct { __m256d val[3]; } npyv_f64x3;

#define npyv_nlanes_u8  32
#define npyv_nlanes_s8  32
#define npyv_nlanes_u16 16
#define npyv_nlanes_s16 16
#define npyv_nlanes_u32 8
#define npyv_nlanes_s32 8
#define npyv_nlanes_u64 4
#define npyv_nlanes_s64 4
#define npyv_nlanes_f32 8
#define npyv_nlanes_f64 4


#include "memory.h"
#include "misc.h"
#include "reorder.h"
#include "operators.h"
#include "conversion.h"
#include "arithmetic.h"
#include "math.h"
