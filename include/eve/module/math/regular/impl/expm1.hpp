//==================================================================================================
/*
  EVE - Expressive Vector Engine
  Copyright : EVE Project Contributors
  SPDX-License-Identifier: BSL-1.0
*/
//==================================================================================================
#pragma once

#include <eve/module/core.hpp>
#include <eve/module/math/regular/horner.hpp>
#include <eve/module/core/detail/generic/horn.hpp>
#include <eve/module/math/constant/log_2.hpp>
#include <eve/module/math/constant/maxlog.hpp>

namespace eve::detail
{
  template<floating_ordered_value T, decorator D>
  EVE_FORCEINLINE constexpr T
  expm1_(EVE_SUPPORTS(cpu_), D const&, T xx) noexcept
  requires(is_one_of<D>(types<regular_type, pedantic_type> {}))
  {
    if constexpr( has_native_abi_v<T> )
    {
      using elt_t       = element_type_t<T>;
      using i_t         = as_integer_t<T>;
      const T Log_2hi   = ieee_constant<0x1.6300000p-1f, 0x1.62e42fee00000p-1>(eve::as<T>{});
      const T Log_2lo   = ieee_constant<-0x1.bd01060p-13f, 0x1.a39ef35793c76p-33>(eve::as<T>{});
      const T Invlog_2  = ieee_constant<0x1.7154760p+0f, 0x1.71547652b82fep+0>(eve::as<T>{});
      T       k         = nearest(Invlog_2 * xx);
      auto    xlelogeps = xx <= logeps(eve::as(xx));
      auto    xgemaxlog = xx >= maxlog(eve::as(xx));
      if constexpr( scalar_value<T> )
      {
        if( is_eqz(xx) || is_nan(xx) ) return xx;
        if( xgemaxlog ) return inf(eve::as<T>());
        if( xlelogeps ) return mone(eve::as<T>());
      }
      if constexpr( std::is_same_v<elt_t, float> )
      {
        T x    = fnma(k, Log_2hi, xx);
        x      = fnma(k, Log_2lo, x);
        T hx   = x * half(eve::as<T>());
        T hxs  = x * hx;
        T r1   =
          eve::reverse_horner(hxs, T(0x1.000000p+0f), T(-0x1.1110fep-5f), T(0x1.9edb68p-10f))
          ;
        T t    = fnma(r1, hx, T(3));
        T e    = hxs * ((r1 - t) / (T(6) - x * t));
        e      = fms(x, e, hxs);
        i_t ik = int_(k);
        T   two2mk =
          bit_cast((maxexponent(eve::as<T>()) - ik) << nbmantissabits(eve::as<elt_t>()), as<T>());
        k = oneminus(two2mk) - (e - x);
        k = D()(ldexp)(k, ik);
      }
      else if constexpr( std::is_same_v<elt_t, double> )
      {
        T hi   = fnma(k, Log_2hi, xx);
        T lo   = k * Log_2lo;
        T x    = hi - lo;
        T hxs  = sqr(x) * half(eve::as<T>());
        T r1   =
          eve::reverse_horner(hxs, T(0x1.0000000000000p+0), T(-0x1.11111111110f4p-5), T(0x1.a01a019fe5585p-10)
                             , T(-0x1.4ce199eaadbb7p-14), T(0x1.0cfca86e65239p-18), T(-0x1.afdb76e09c32dp-23))
          ;
        T t    = T(3) - r1 * half(eve::as<T>()) * x;
        T e    = hxs * ((r1 - t) / (T(6) - x * t));
        T c    = (hi - x) - lo;
        e      = (x * (e - c) - c) - hxs;
        i_t ik = int_(k);
        T   two2mk =
          bit_cast((maxexponent(eve::as<T>()) - ik) << nbmantissabits(eve::as<T>()), as<T>());
        T ct1 = oneminus(two2mk) - (e - x);
        T ct2 = inc((x - (e + two2mk)));
        k     = if_else((k < T(20)), ct1, ct2);
        k     = D()(ldexp)(k, ik);
      }
      if constexpr( simd_value<T> )
      {
        k = if_else(xgemaxlog, inf(eve::as<T>()), k);
        k = if_else(is_eqz(xx) || is_nan(xx), xx, k);
        k = if_else(xlelogeps, eve::mone, k);
      }
      return k;
    }
    else return apply_over(expm1, xx);
  }

  template<floating_ordered_value T>
  EVE_FORCEINLINE constexpr T
  expm1_(EVE_SUPPORTS(cpu_), T const& x) noexcept
  {
    return expm1(regular_type(), x);
  }

// -----------------------------------------------------------------------------------------------
// Masked case
  template<conditional_expr C, value U>
  EVE_FORCEINLINE auto
  expm1_(EVE_SUPPORTS(cpu_), C const& cond, U const& t) noexcept
  {
    return mask_op(cond, eve::expm1, t);
  }
}
