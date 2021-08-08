//==================================================================================================
/**
  EVE - Expressive Vector Engine
  Copyright : EVE Contributors & Maintainers
  SPDX-License-Identifier: MIT
**/
//==================================================================================================
#pragma once

#include <eve/detail/implementation.hpp>
#include <type_traits>
#include <eve/concept/value.hpp>

namespace eve::detail
{
  // -----------------------------------------------------------------------------------------------
  // 128 bits implementation
  template<integral_real_scalar_value T, integral_real_scalar_value I, typename N>
  EVE_FORCEINLINE wide<T, N> rotl_(EVE_SUPPORTS(avx_), wide<T, N> a0, I a1) noexcept
    requires std::same_as<abi_t<T, N>, x86_128_> && supports_xop
  {
           if constexpr(sizeof(T) == 8) return _mm_rot_epi64(a0,a1);
      else if constexpr(sizeof(T) == 4) return _mm_rot_epi32(a0,a1);
      else if constexpr(sizeof(T) == 2) return _mm_rot_epi16(a0,a1);
      else if constexpr(sizeof(T) == 1) return _mm_rot_epi8(a0,a1);
  }

  // -----------------------------------------------------------------------------------------------
  // avx512 implementation
  template<integral_real_scalar_value T, integral_real_scalar_value I, typename N>
  EVE_FORCEINLINE wide<T, N> rotl_(EVE_SUPPORTS(avx512_), wide<T, N> a0,  wide<I, N> a1) noexcept
  {
    constexpr auto c = categorize<wide<T, N>>();
    if constexpr(c == category::int32x16 )  return _mm512_rolv_epi32(a0, a1);
    else  if constexpr(c == category::int64x8  )  return _mm512_rolv_epi64(a0, a1);
    else  if constexpr(c == category::int32x8  )  return _mm256_rolv_epi32(a0, a1);
    else  if constexpr(c == category::int64x4  )  return _mm256_rolv_epi64(a0, a1);
    else  if constexpr(c == category::int32x4  )  return _mm_rolv_epi32(a0, a1);
    else  if constexpr(c == category::int64x2  )  return _mm_rolv_epi64(a0, a1);
    else  return rotl_(EVE_RETARGET(cpu_), a0, a1);
  }

  // -----------------------------------------------------------------------------------------------
  // masked avx512 implementation
  template<conditional_expr C,integral_real_scalar_value T, integral_real_scalar_value I, typename N>
  EVE_FORCEINLINE wide<T, N> rotl_(EVE_SUPPORTS(avx512_), C const &cx, wide<T, N> a0,  wide<I, N> a1) noexcept
  {
    constexpr auto c = categorize<wide<T, N>>();

    if constexpr( C::is_complete || abi_t<T, N>::is_wide_logical )
    {
      return rotl_(EVE_RETARGET(cpu_),cx,a0, a1);
    }
    else
    {
      auto src  = alternative(cx,a0,as<wide<T, N>>{});
      auto m    = expand_mask(cx,as<wide<T, N>>{}).storage().value;

      if constexpr(c == category::int32x16 )  return _mm512_rolv_epi32(src, m, a0, a1);
      else  if constexpr(c == category::int64x8  )  return _mm512_rolv_epi64(src, m, a0, a1);
      else  if constexpr(c == category::int32x8  )  return _mm256_rolv_epi32(src, m, a0, a1);
      else  if constexpr(c == category::int64x4  )  return _mm256_rolv_epi64(src, m, a0, a1);
      else  if constexpr(c == category::int32x4  )  return _mm_rolv_epi32(src, m, a0, a1);
      else  if constexpr(c == category::int64x2  )  return _mm_rolv_epi64(src, m, a0, a1);
      else  return rotl_(EVE_RETARGET(cpu_), cx, a0, a1);

    }
  }
}
