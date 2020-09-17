//==================================================================================================
/**
  EVE - Expressive Vector Engine
  Copyright 2020 Joel FALCOU
  Copyright 2020 Jean-Thierry LAPRESTE

  Licensed under the MIT License <http://opensource.org/licenses/MIT>.
  SPDX-License-Identifier: MIT
**/
//==================================================================================================
#pragma once

#include <eve/detail/overload.hpp>
#include <eve/detail/abi.hpp>
#include <eve/forward.hpp>
#include <type_traits>
#include <eve/concept/value.hpp>

namespace eve::detail
{
  template<real_scalar_value T, typename N>
  EVE_FORCEINLINE wide<T, N, neon64_> bit_ornot_(EVE_SUPPORTS(neon128_),
                                                     wide<T, N, neon64_> const &v0,
                                                     wide<T, N, neon64_> const &v1) noexcept
  {
    using in_t = typename wide<T, N, neon64_>::storage_type;

// Dispatch to sub-functions

         if constexpr(std::is_same_v<in_t, float32x2_t>)
            return vreinterpret_f32_u32(vorn_u32(vreinterpret_u32_f32(v0), vreinterpret_u32_f32(v1)));
#if defined(__aarch64__)
    else if constexpr(std::is_same_v<in_t, float64x1_t>)
            return vreinterpret_f64_u64(vorn_u64(vreinterpret_u64_f64(v0), vreinterpret_u64_f64(v1)));
#endif

    else if constexpr(std::is_same_v<in_t, int64x1_t>)  return vorn_s64(v0, v1);
    else if constexpr(std::is_same_v<in_t, int32x2_t>)  return vorn_s32(v0, v1);
    else if constexpr(std::is_same_v<in_t, int16x4_t>)  return vorn_s16(v0, v1);
    else if constexpr(std::is_same_v<in_t, int8x8_t>)   return vorn_s8(v0, v1);
    else if constexpr(std::is_same_v<in_t, uint64x1_t>) return vorn_u64(v0, v1);
    else if constexpr(std::is_same_v<in_t, uint32x2_t>) return vorn_u32(v0, v1);
    else if constexpr(std::is_same_v<in_t, uint16x4_t>) return vorn_u16(v0, v1);
    else if constexpr(std::is_same_v<in_t, uint8x8_t>)  return vorn_u8(v0, v1);
  }

  template<real_scalar_value T, typename N>
  EVE_FORCEINLINE wide<T, N, neon128_> bit_ornot_(EVE_SUPPORTS(neon128_),
                                                      wide<T, N, neon128_> const &v0,
                                                      wide<T, N, neon128_> const &v1) noexcept
  {
    using in_t = typename wide<T, N, neon128_>::storage_type;

// Dispatch to sub-functions
         if constexpr(std::is_same_v<in_t, float32x4_t>)
            return vreinterpretq_f32_u32(vornq_u32(vreinterpretq_u32_f32(v0), vreinterpretq_u32_f32(v1)));
#if defined(__aarch64__)
    else if constexpr(std::is_same_v<in_t, float64x2_t>)
            return vreinterpretq_f64_u64(vornq_u64(vreinterpretq_u64_f64(v0), vreinterpretq_u64_f64(v1)));
#endif


    else if constexpr(std::is_same_v<in_t, int64x2_t>)  return vornq_s64(v0, v1);
    else if constexpr(std::is_same_v<in_t, int32x4_t>)  return vornq_s32(v0, v1);
    else if constexpr(std::is_same_v<in_t, int16x8_t>)  return vornq_s16(v0, v1);
    else if constexpr(std::is_same_v<in_t, int8x16_t>)  return vornq_s8(v0, v1);
    else if constexpr(std::is_same_v<in_t, uint64x2_t>) return vornq_u64(v0, v1);
    else if constexpr(std::is_same_v<in_t, uint32x4_t>) return vornq_u32(v0, v1);
    else if constexpr(std::is_same_v<in_t, uint16x8_t>) return vornq_u16(v0, v1);
    else if constexpr(std::is_same_v<in_t, uint8x16_t>) return vornq_u8(v0, v1);
  }
}
