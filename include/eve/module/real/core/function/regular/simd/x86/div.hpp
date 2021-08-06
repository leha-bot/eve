//==================================================================================================
/**
  EVE - Expressive Vector Engine
  Copyright : EVE Contributors & Maintainers
  SPDX-License-Identifier: MIT
**/
//==================================================================================================
#pragma once

#include <eve/concept/value.hpp>
#include <eve/detail/implementation.hpp>


namespace eve::detail
{
  // -----------------------------------------------------------------------------------------------
  // Masked case
  template<conditional_expr C, floating_real_scalar_value T, typename N>
  EVE_FORCEINLINE
  wide<T, N> div_(EVE_SUPPORTS(sse2_), C const &cx, wide<T, N> const &v, wide<T, N> const &w) noexcept
  requires x86_abi<abi_t<T, N>>
  {
    constexpr auto c = categorize<wide<T, N>>();

    if constexpr( C::is_complete || abi_t<T, N>::is_wide_logical )
    {
      return div_(EVE_RETARGET(cpu_),cx,v,w);
    }
    else
    {
      auto src  = alternative(cx,v,as<wide<T, N>>{});
      auto m    = expand_mask(cx,as<wide<T, N>>{}).storage().value;

            if constexpr(c == category::float32x16) return _mm512_mask_div_ps   (src,m,v,w);
      else  if constexpr(c == category::float64x8 ) return _mm512_mask_div_pd   (src,m,v,w);
      else  {
        return div_(EVE_RETARGET(cpu_),cx,v,w);
      }
    }
  }

  // -----------------------------------------------------------------------------------------------
  // Rouding case
  template<decorator D, floating_real_scalar_value T, typename N>
  EVE_FORCEINLINE
  wide<T, N> div_(EVE_SUPPORTS(avx512_)
                , D const &, wide<T, N> const &v, wide<T, N> const &w) noexcept
  requires(is_one_of<D>(types<toward_zero_type, downward_type, to_nearest_type, upward_type> {}))
  {
    constexpr auto c = categorize<wide<T, N>>();

          if constexpr(c == category::float32x16) return _mm512_div_round_ps(v,w,D::base_type::value);
    else  if constexpr(c == category::float64x8 ) return _mm512_div_round_pd(v,w,D::base_type::value);
    else return div_(EVE_RETARGET(cpu_), D(), v, w);

  }

  // -----------------------------------------------------------------------------------------------
  // masked Rouding case
  template<decorator D, conditional_expr C, floating_real_scalar_value T, typename N>
  EVE_FORCEINLINE
  wide<T, N> div_(EVE_SUPPORTS(avx512_), C const &cx
                , D const &, wide<T, N> const &v, wide<T, N> const &w) noexcept
  requires(is_one_of<D>(types<toward_zero_type, downward_type, to_nearest_type, upward_type> {}))
  {
    constexpr auto c = categorize<wide<T, N>>();

    if constexpr( C::is_complete || abi_t<T, N>::is_wide_logical )
    {
      return div_(EVE_RETARGET(cpu_),cx,D(),v,w);
    }
    else
    {
      auto src  = alternative(cx,v,as<wide<T, N>>{});
      auto m    = expand_mask(cx,as<wide<T, N>>{}).storage().value;

            if constexpr(c == category::float32x16) return _mm512_mask_div_round_ps(src,m,v,w,D::base_type::value);
      else  if constexpr(c == category::float64x8 ) return _mm512_mask_div_round_pd(src,m,v,w,D::base_type::value);
      else return div_(EVE_RETARGET(cpu_), cx, D(), v, w);
    }
  }
}
