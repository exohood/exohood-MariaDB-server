#include <stddef.h>
#include <stdint.h>
#include <my_global.h>

#ifdef _MSC_VER
# include <intrin.h>
# define USE_VPCLMULQDQ /* nothing */
#else
# include <cpuid.h>
# if __GNUC__ < 5 && !defined __clang__
/* the headers do not really work in GCC before version 5 */
#  define _mm_crc32_u8(crc,data) __builtin_ia32_crc32qi(crc,data)
#  define _mm_crc32_u32(crc,data) __builtin_ia32_crc32si(crc,data)
#  define _mm_crc32_u64(crc,data) __builtin_ia32_crc32di(crc,data)
# else
#  if SIZEOF_SIZE_T != 8
#  elif __GNUC__ >= 11 || (defined __clang_major__ && __clang_major__ >= 8)
#   define TARGET_VPCLMULQDQ \
  "pclmul,avx512f,avx512dq,avx512bw,avx512vl,vpclmulqdq"
#   define USE_VPCLMULQDQ __attribute__((target(TARGET_VPCLMULQDQ)))
#  endif
# endif
#endif

#ifdef USE_VPCLMULQDQ
# include <immintrin.h>
#endif

extern "C" unsigned crc32c_sse42(unsigned crc, const void* buf, size_t size);

constexpr uint32_t cpuid_ecx_SSE42= 1U << 20;
constexpr uint32_t cpuid_ecx_SSE42_AND_PCLMUL= cpuid_ecx_SSE42 | 1U << 1;

static uint32_t cpuid_ecx()
{
#ifdef __GNUC__
  uint32_t reax= 0, rebx= 0, recx= 0, redx= 0;
  __cpuid(1, reax, rebx, recx, redx);
  return recx;
#elif defined _MSC_VER
  int regs[4];
  __cpuid(regs, 1);
  return regs[2];
#else
# error "unknown compiler"
#endif
}

typedef unsigned (*my_crc32_t)(unsigned, const void *, size_t);
extern "C" unsigned int crc32_pclmul(unsigned int, const void *, size_t);
extern "C" unsigned int crc32c_3way(unsigned int, const void *, size_t);

#ifdef USE_VPCLMULQDQ
/** table of constants corresponding to a CRC polynomial up to degree 32 */
struct crc32_tab
{
  alignas(64) const
  uint64_t b2048[2], b1024[2], b896[2], b768[2], b640[2], b512[2], b384[2],
    b256[2], b128[2], b64[2], b32[2];
};

/** ISO 3309 CRC-32 (polynomial 0x04C11DB7) */
static const crc32_tab crc32_const= {
  { 0x00000000e95c1271, 0x00000000ce3371cb },
  { 0x00000000910eeec1, 0x0000000033fff533 },
  { 0x000000000cbec0ed, 0x0000000031f8303f },
  { 0x0000000057c54819, 0x00000000df068dc2 },
  { 0x00000000ae0b5394, 0x000000001c279815 },
  { 0x000000001d9513d7, 0x000000008f352d95 },
  { 0x00000000af449247, 0x000000003db1ecdc },
  { 0x0000000081256527, 0x00000000f1da05aa },
  { 0x00000000ccaa009e, 0x00000000ae689191 },
  { 0x00000000ccaa009e, 0x00000000b8bc6765 },
  { 0x00000001f7011640, 0x00000001db710640 }
};

/** Castagnoli CRC-32C (polynomial 0x1EDC6F41) */
static const crc32_tab crc32c_const= {
  { 0x000000004ef6a711, 0x00000000fa374b2e },
  { 0x00000000e78dbf1d, 0x000000005a47b20d },
  { 0x0000000079d09793, 0x00000000da9c52d0 },
  { 0x00000000ac594d98, 0x000000007def8667 },
  { 0x0000000038f8236c, 0x000000009a6aeb31 },
  { 0x00000000aa97d41d, 0x00000000a6955f31 },
  { 0x00000000e6957b4d, 0x00000000aa5eec4a },
  { 0x0000000059a3508a, 0x000000007bba6798 },
  { 0x0000000018571d18, 0x000000006503ea99 },
  { 0xd7a0166500000000, 0x3aab457600000000 },
  { 0x000000011f91caf6, 0x000000011edc6f41 }
};

static constexpr uint8_t mm_XOR= 0x96;

USE_VPCLMULQDQ
/** 3-ary exclusive or */
static inline __m128i xor3_128(__m128i a, __m128i b, __m128i c)
{
  return _mm_ternarylogic_epi64(a, b, c, mm_XOR);
}

USE_VPCLMULQDQ
/** 3-ary exclusive or */
static inline __m512i xor3_512(__m512i a, __m512i b, __m512i c)
{
  return _mm512_ternarylogic_epi64(a, b, c, mm_XOR);
}

USE_VPCLMULQDQ
/** Load 64 bytes */
static inline __m512i load512(__m512i S, const char *buf)
{
  return _mm512_shuffle_epi8(_mm512_loadu_epi8(buf), S);
}

USE_VPCLMULQDQ
/** Load 16 bytes */
static inline __m128i load128(__m512i S, const char *buf)
{
  return _mm_shuffle_epi8(_mm_loadu_epi64(buf), _mm512_castsi512_si128(S));
}

/** Combine 512 data bits with CRC */
#define combine512(a, tab, b)                           \
  xor3_512(b, _mm512_clmulepi64_epi128(a, tab, 0x11),   \
  _mm512_clmulepi64_epi128(a, tab, 0))

#define xor512(a, b) _mm512_xor_epi64(a, b)
#define xor256(a, b) _mm256_xor_epi64(a, b)
#define xor128(a, b) _mm_xor_epi64(a, b)
#define and128(a, b) _mm_and_si128(a, b)

static const __m128i mask1=
  _mm_set_epi64x(0x8080808080808080, 0x8080808080808080);
alignas(64) static const __mmask16 size_mask[16]= {
  0x0001, 0x0003, 0x0007, 0x000f, 0x001f, 0x003f, 0x007f, 0x00ff,
  0x01ff, 0x03ff, 0x07ff, 0x0fff, 0x1fff, 0x3fff, 0x5fff, 0x7fff
};

USE_VPCLMULQDQ ATTRIBUTE_NOINLINE
static unsigned crc32_avx512(unsigned crc, const char *buf, size_t size,
                             const crc32_tab &tab)
{
  const __m512i S= _mm512_broadcast_i32x4(mask1),
    crc_in=
    _mm512_bslli_epi128(_mm512_castsi128_si512(_mm_cvtsi32_si128(crc)), 12),
    b512= _mm512_broadcast_i32x4(*reinterpret_cast<const __m128i*>(&tab.b512));
  __m128i crc_out;
  __m512i m0;

  if (size >= 256)
  {
    m0= xor512(load512(S, buf), crc_in);
    __m512i m4= xor512(load512(S, buf + 64), crc_in);

    const __m512i b1024=
      _mm512_broadcast_i32x4(*reinterpret_cast<const __m128i*>(&tab.b1024));
    size-= 256;
    if (size >= 256)
    {
      __m512i
        m7= _mm512_shuffle_epi8(_mm512_loadu_epi8(buf + 128), S),
        m8= _mm512_shuffle_epi8(_mm512_loadu_epi8(buf + 192), S);
      const __m512i b2048=
        _mm512_broadcast_i32x4(*reinterpret_cast<const __m128i*>(&tab.b2048));
      do
      {
        buf+= 256;
        m0= combine512(m0, b2048, load512(S, buf));
        m4= combine512(m4, b2048, load512(S, buf + 64));
        m7= combine512(m7, b2048, load512(S, buf + 128));
        m8= combine512(m8, b2048, load512(S, buf + 192));
        size-= 256;
      }
      while (ssize_t(size) >= 0);

      buf+= 256;
      m0= combine512(m0, b1024, m7);
      m4= combine512(m8, b1024, m4);
      size+= 128;
    }
    else
    {
      do
      {
        buf+= 128;
        m0= combine512(m0, b1024, load512(S, buf));
        m4= combine512(m4, b1024, load512(S, buf + 64));
        size-= 128;
      } while (ssize_t(size) >= 0);

      buf+= 128;
    }

    if (ssize_t(size) >= -64)
    {
      size+= 128;
      m0= combine512(m0, b512, m4);
      goto fold_64_B_loop;
    }
    else
    {
      {
        const __m512i
          b896= _mm512_loadu_epi8(&tab.b896),
          b384= _mm512_loadu_epi8(&tab.b384);

        crc_out= _mm512_extracti64x2_epi64(m4, 3);
        __m512i m1;
        m1= xor3_512(_mm512_clmulepi64_epi128(m0, b896, 0),
                     _mm512_clmulepi64_epi128(m0, b896, 0x11),
                     _mm512_clmulepi64_epi128(m4, b384, 0));
        m1= xor3_512(m1, _mm512_clmulepi64_epi128(m4, b384, 0x11),
                     _mm512_castsi128_si512(crc_out));

        __m256i m8=
          _mm512_castsi512_si256(_mm512_shuffle_i64x2(m1, m1, 0b01001110));
        m8= xor256(m8, _mm512_castsi512_si256(m1));
        crc_out= xor128(_mm256_extracti64x2_epi64(m8, 1),
                        _mm256_castsi256_si128(m8));
      }

      size+= 128 - 16;
      goto final_reduction;
    }
  }

  __m128i b;

  // less_than_256
  if (size >= 32)
  {
    if (size >= 64)
    {
      m0= xor512(load512(S, buf), crc_in);

      buf+= 64;
      size-= 64;

      if (size >= 64)
      {
      fold_64_B_loop:
        do
        {
          m0= combine512(m0, b512, load512(S, buf));
          buf+= 64;
          size-= 64;
        }
        while (size >= 64);
      }

      // reduce_64B:
      const __m512i b384= _mm512_loadu_epi8(&tab.b384);
      __m512i crc512=
        xor3_512(_mm512_clmulepi64_epi128(m0, b384, 0x11),
                 _mm512_clmulepi64_epi128(m0, b384, 0),
                 _mm512_castsi128_si512(_mm512_extracti64x2_epi64(m0, 3)));
      crc512= xor512(crc512, _mm512_shuffle_i64x2(crc512, crc512, 0b01001110));
      const __m256i crc256= _mm512_castsi512_si256(crc512);
      crc_out= xor128(_mm256_extracti64x2_epi64(crc256, 1),
                      _mm256_castsi256_si128(crc256));
      size-= 16;
    }
    else
    {
      // less_than_64
      crc_out= xor128(load128(S, buf), _mm512_castsi512_si128(crc_in));
      buf+= 16;
      size-= 32;
    }

  final_reduction:
    b= *reinterpret_cast<const __m128i*>(&tab.b128);

    while (ssize_t(size) >= 0)
    {
      // reduction_loop_16B:
      __m256i m8=
        _mm256_castsi128_si256(_mm_clmulepi64_si128(crc_out, b, 0x11));
      crc_out= _mm_clmulepi64_si128(crc_out, b, 0);
      crc_out= xor128(crc_out, _mm256_castsi256_si128(m8));
      crc_out= xor128(crc_out, load128(S, buf));
      buf+= 16;
      size-= 16;
    }
    // final_reduction_for_128
    alignas(16) static const uint64_t shuffle[4]=
      { 0x8786858483828100, 0x8f8e8d8c8b8a8988,
        0x0706050403020100, 0x000e0d0c0b0a0908 };

    size+= 16;
    if (size)
    {
    get_last_two_xmms:
      __m128i m1= load128(S, buf + (size - 16));
      __m128i m0= _mm_loadu_epi64(reinterpret_cast<const char*>(shuffle) -
                                  (size - 16));
      __m128i m2= _mm_shuffle_epi8(crc_out, m0);
      crc_out= xor128(crc_out, _mm512_castsi512_si128(S));
      crc_out= _mm_shuffle_epi8(crc_out, m0);
      m1= _mm_blendv_epi8(m1, m2, m0);
      crc_out= _mm_clmulepi64_si128(crc_out, b, 0);
      crc_out= xor3_128(crc_out, m1, _mm_clmulepi64_si128(crc_out, b, 0x11));
    }

    __m128i m0;
  done_128:
    b= *reinterpret_cast<const __m128i*>(&tab.b64);
    m0= _mm_slli_si128(crc_out, 8);
    crc_out= _mm_clmulepi64_si128(crc_out, b, 0x01);
    crc_out= xor128(crc_out, m0);
    m0= and128(crc_out, _mm_set_epi64x(~0ULL, ~0U));
    crc_out= _mm_srli_si128(crc_out, 12);
    crc_out= _mm_clmulepi64_si128(crc_out, b, 0x10);
    crc_out= xor128(crc_out, m0);

  barrett:
    b= *reinterpret_cast<const __m128i*>(&tab.b32);
    m0= crc_out;
    crc_out= _mm_clmulepi64_si128(crc_out, b, 0x01);
    crc_out= _mm_slli_si128(crc_out, 4);
    crc_out= _mm_clmulepi64_si128(crc_out, b, 0x11);
    crc_out= _mm_slli_si128(crc_out, 4);
    crc_out= xor128(crc_out, m0);
    return _mm_extract_epi32(crc_out, 1);
  }
  else
  {
    // less_than_32
    if (likely(size > 0))
    {
      if (size > 16)
      {
        crc_out= load128(S, buf);
        buf+= 16;
        size-= 16;
        b= *reinterpret_cast<const __m128i*>(&tab.b128);
        goto get_last_two_xmms;
      }
      else if (size < 16)
      {
        crc_out= _mm_maskz_loadu_epi8(size_mask[size - 1], buf);
        crc_out= _mm_shuffle_epi8(crc_out, _mm512_castsi512_si128(S));
        crc_out= xor128(crc_out, _mm512_castsi512_si128(crc_in));

        if (size >= 4)
        {
          alignas(16) static const uint64_t shift[4]=
          { 0x8786858483828100, 0x8f8e8d8c8b8a8988,
            0x0706050403020100, 0x000e0d0c0b0a0908 };
          crc_out= _mm_shuffle_epi8
            (crc_out, xor128(_mm_loadu_epi64(reinterpret_cast<const char*>
                                             (shift) + 16 - size), mask1));
          goto done_128;
        }
        else
        {
          // only_less_than_4:
          alignas(16) static const int8_t shift[17]=
          {5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, -1, -1, -1, -1, -1, -1};
          crc_out= _mm_shuffle_epi8
            (crc_out, _mm_loadu_epi64(reinterpret_cast<const char*>(shift) +
                                      3 - size));
          goto barrett;
        }
      }
      else
      {
        crc_out= xor128(load128(S, buf), _mm512_castsi512_si128(crc_in));
        goto done_128;
      }
    }

    return crc;
  }
}

static ATTRIBUTE_NOINLINE bool have_vpclmulqdq()
{
# ifdef _MSC_VER
  int regs[4];
  __cpuidex(regs, 7, 0);
  uint32_t ebx= regs[1], ecx= regs[2];
# else
  uint32_t eax= 0, ebx= 0, ecx= 0, edx= 0;
  __cpuid_count(7, 0, eax, ebx, ecx, edx);
# endif
  return ecx & 1U<<10/*VPCLMULQDQ*/ &&
    !(~ebx & ((1U<<16/*AVX512F*/ | 1U<<17/*AVX512DQ*/ |
               1U<<30/*AVX512BW*/ | 1U<<31/*AVX512VL*/)));
}

static unsigned crc32_vpclmulqdq(unsigned crc, const void *buf, size_t size)
{
  return crc32_avx512(crc, static_cast<const char*>(buf), size, crc32_const);
}

static unsigned crc32c_vpclmulqdq(unsigned crc, const void *buf, size_t size)
{
  return crc32_avx512(crc, static_cast<const char*>(buf), size, crc32c_const);
}
#else
static constexpr bool have_vpclmulqdq() { return false; }
static constexpr my_crc32_t crc32_vpclmulqdq= crc32_pclmul;
static constexpr my_crc32_t crc32c_vpclmulqdq= crc32c_3way;
#endif

extern "C" my_crc32_t crc32_pclmul_enabled(void)
{
  if (~cpuid_ecx() & cpuid_ecx_SSE42_AND_PCLMUL)
    return nullptr;
  if (have_vpclmulqdq())
    return crc32_vpclmulqdq;
  return crc32_pclmul;
}

extern "C" my_crc32_t crc32c_x86_available(void)
{
  if (have_vpclmulqdq())
    return crc32c_vpclmulqdq;
#if SIZEOF_SIZE_T == 8
  switch (cpuid_ecx() & cpuid_ecx_SSE42_AND_PCLMUL) {
  case cpuid_ecx_SSE42_AND_PCLMUL:
    return crc32c_3way;
  case cpuid_ecx_SSE42:
    return crc32c_sse42;
  }
#else
  if (cpuid_ecx() & cpuid_ecx_SSE42)
    return crc32c_sse42;
#endif
  return nullptr;
}

extern "C" const char *crc32c_x86_impl(my_crc32_t c)
{
#ifdef USE_VPCLMULQDQ
  if (c == crc32c_vpclmulqdq)
    return "Using AVX512 instructions";
#endif
#if SIZEOF_SIZE_T == 8
  if (c == crc32c_3way)
    return "Using crc32 + pclmulqdq instructions";
#endif
  if (c == crc32c_sse42)
    return "Using SSE4.2 crc32 instructions";
  return nullptr;
}
