//==================================================================================================
/**
  EVE - Expressive Vector Engine
  Copyright 2020 Joel FALCOU
  Copyright 2020 Jean-Thierry LAPRESTE

  Licensed under the MIT License <http://opensource.org/licenses/MIT>.
  SPDX-License-Identifier: MIT
**/
//==================================================================================================
#ifndef EVE_MODULE_CORE_FUNCTION_SIMD_X86_BIT_OR_HPP_INCLUDED
#define EVE_MODULE_CORE_FUNCTION_SIMD_X86_BIT_OR_HPP_INCLUDED

#include <eve/detail/overload.hpp>
#include <eve/detail/abi.hpp>
#include <eve/forward.hpp>
#include <type_traits>

namespace eve::detail
{
  // -----------------------------------------------------------------------------------------------
  // 128 bits implementation
  template<typename T, typename N>
  EVE_FORCEINLINE wide<T, N, sse_>
                  bit_or_(EVE_SUPPORTS(sse2_), wide<T, N, sse_> const &v0, wide<T, N, sse_> const &v1) noexcept
  {
    if constexpr(std::is_same_v<T, float>)
      return _mm_or_ps(v0, v1);
    else if constexpr(std::is_same_v<T, double>)
      return _mm_or_pd(v0, v1);
    else if constexpr(std::is_integral_v<T>)
      return _mm_or_si128(v0, v1);
  }

  // -----------------------------------------------------------------------------------------------
  // 256 bits implementation
  template<typename T, typename N>
  EVE_FORCEINLINE wide<T, N, avx_>
                  bit_or_(EVE_SUPPORTS(avx_), wide<T, N, avx_> const &v0, wide<T, N, avx_> const &v1) noexcept
  {
    if constexpr(std::is_same_v<T, float>)
      return _mm256_or_ps(v0, v1);
    else if constexpr(std::is_same_v<T, double>)
      return _mm256_or_pd(v0, v1);
    else if constexpr(std::is_integral_v<T>)
    {
      if constexpr(current_api >= avx2) { return _mm256_or_si256(v0, v1); }
      else
      {
        return _mm256_castps_si256(_mm256_or_ps(_mm256_castsi256_ps(v0), _mm256_castsi256_ps(v1)));
      }
    }
  }
}

#endif
