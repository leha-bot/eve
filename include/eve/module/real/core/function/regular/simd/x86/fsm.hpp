//==================================================================================================
/**
  EVE - Expressive Vector Engine
  Copyright : EVE Contributors & Maintainers
  SPDX-License-Identifier: MIT
**/
//==================================================================================================
#pragma once

#include <eve/concept/value.hpp>
#include <eve/detail/category.hpp>
#include <eve/detail/implementation.hpp>
#include <type_traits>

namespace eve::detail
{
  // -----------------------------------------------------------------------------------------------
  // Masked case
  template<conditional_expr C, real_scalar_value T, typename N>
  EVE_FORCEINLINE
  wide<T, N> fsm_(EVE_SUPPORTS(sse2_), C const &cx
                 , wide<T, N> const &v
                 , wide<T, N> const &w
                 , wide<T, N> const &x) noexcept
  requires x86_abi<abi_t<T, N>>
  {
    constexpr auto c = categorize<wide<T, N>>();

    if constexpr( C::is_complete || abi_t<T, N>::is_wide_logical )
    {
      return fsm_(EVE_RETARGET(cpu_),cx,v,w,x);
    }
    else
    {
      auto m    = expand_mask(cx,as<wide<T, N>>{}).storage().value;

      if constexpr(!C::has_alternative)
      {
              if constexpr(c == category::float32x16) return _mm512_mask3_fmsub_ps(w,x,v,m);
        else  if constexpr(c == category::float64x8 ) return _mm512_mask3_fmsub_pd(w,x,v,m);
        else  if constexpr(c == category::float32x8 ) return _mm256_mask3_fmsub_ps(w,x,v,m);
        else  if constexpr(c == category::float64x4 ) return _mm256_mask3_fmsub_pd(w,x,v,m);
        else  if constexpr(c == category::float32x8 ) return _mm128_mask3_fmsub_ps(w,x,v,m);
        else  if constexpr(c == category::float64x4 ) return _mm128_mask3_fmsub_pd(w,x,v,m);
        else  return fsm_(EVE_RETARGET(cpu_),cx,v,w,x);
      }
      else
      {
        auto src  = alternative(cx,v,as<wide<T, N>>{});
        return fsm_(EVE_RETARGET(cpu_),cx,v,w,x);
      }
    }
  }

 //  // -----------------------------------------------------------------------------------------------
//   // masked Rouding case
//   template<decorator D, conditional_expr C, floating_real_scalar_value T, typename N>
//   EVE_FORCEINLINE
//   wide<T, N> fsm_(EVE_SUPPORTS(avx512_), C const &cx
//                 , D const &, wide<T, N> const &v, wide<T, N> const &w, wide<T, N> const &x) noexcept
//   requires(is_one_of<D>(types<toward_zero_type, downward_type, to_nearest_type, upward_type> {}))
//   {
//     constexpr auto c = categorize<wide<T, N>>();

//     if constexpr( C::is_complete || abi_t<T, N>::is_wide_logical )
//     {
//       return fsm_(EVE_RETARGET(cpu_),cx,D(),v,w,x);
//     }
//     else
//     {
//       auto src  = alternative(cx,v,as<wide<T, N>>{});
//       auto m    = expand_mask(cx,as<wide<T, N>>{}).storage().value;

//             if constexpr(c == category::float32x16) return _mm512_mask3_fmsub_round_ps(v,w,x,m,D::base_type::value);
//       else  if constexpr(c == category::float64x8 ) return _mm512_mask3_fmsub_round_pd(v,w,x,m,D::base_type::value);
//       else return fsm_(EVE_RETARGET(cpu_), cx, D(), v, w, x);
//     }
//  }
}
