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

namespace eve::detail
{
  template<typename T, callable_options O>
  constexpr auto
  log1p_(EVE_REQUIRES(cpu_), O const&, T a0) noexcept
  {
    if constexpr(simd_value<T>)
    {
      if constexpr( has_native_abi_v<T> )
      {
        using elt_t         = element_type_t<T>;
        using uiT           = as_integer_t<T, unsigned>;
        using iT            = as_integer_t<T, signed>;
        const elt_t Log_2hi = ieee_constant<0x1.6300000p-1f, 0x1.62e42fee00000p-1>(eve::as<elt_t>{});
        const elt_t Log_2lo = ieee_constant<-0x1.bd01060p-13f, 0x1.a39ef35793c76p-33>(eve::as<elt_t>{});
        constexpr bool is_avx = current_api == avx;
        if constexpr(is_avx)
        {
          T    uf          = inc(a0);
          auto isnez       = is_nez(uf);
          auto [x, k]      = frexp(uf);
          auto x_lt_sqrthf = (invsqrt_2(eve::as<T>()) > x);
          /* reduce x into [sqrt(2)/2, sqrt(2)] */
          k   = dec[x_lt_sqrthf](k);
          T f = dec(x + if_else(x_lt_sqrthf, x, eve::zero));
          /* correction term ~ log(1+x)-log(u), avoid underflow in c/u */
          T c    = if_else(k >= 2, oneminus(uf - a0), a0 - dec(uf)) / uf;
          T hfsq = half(eve::as<T>()) * sqr(f);
          T s    = f / (2.0f + f);
          T z    = sqr(s);
          T w    = sqr(z);
          T t1, t2;
          if constexpr( std::is_same_v<element_type_t<T>, float> )
          {
            t1 = w *
              eve::reverse_horner(w, T(0x1.999c26p-2f), T(0x1.f13c4cp-3f))
              ;
            t2 = z *
              eve::reverse_horner(w, T(0x1.555554p-1f), T(0x1.23d3dcp-2f))
              ;
          }
          else if constexpr( std::is_same_v<element_type_t<T>, double> )
          {
            t1 = w *
              eve::reverse_horner(w, T(0x1.999999997fa04p-2), T(0x1.c71c51d8e78afp-3), T(0x1.39a09d078c69fp-3))
              ;
            t2 = z*eve::reverse_horner(w, T(0x1.5555555555593p-1), T(0x1.2492494229359p-2)
                                      , T(0x1.7466496cb03dep-3), T(0x1.2f112df3e5244p-3));
          }
          T R = t2 + t1;
          T r = fma(k, Log_2hi, ((fma(s, (hfsq + R), k * Log_2lo + c) - hfsq) + f));
          T zz;
          if constexpr( eve::platform::supports_infinites )
          {
            zz = if_else(
              isnez, if_else(a0 == inf(eve::as<T>()), inf(eve::as<T>()), r), minf(eve::as<T>()));
          }
          else { zz = if_else(isnez, r, minf(eve::as<T>())); }
          return if_else(is_ngez(uf), eve::allbits, zz);
        }
        else
        {
          T           uf      = inc(a0);
          auto        isnez   = is_nez(uf);
          if constexpr( std::is_same_v<elt_t, float> )
          {
            uiT iu = bit_cast(uf, as<uiT>());
            iu += 0x3f800000 - 0x3f3504f3;
            iT k = bit_cast(iu >> 23, as<iT>()) - 0x7f;
            /* correction term ~ log(1+x)-log(u), avoid underflow in c/u */
            T c = if_else(k < 25, if_else(k >= 2, oneminus(uf - a0), a0 - dec(uf)), zero);
            if( eve::any(eve::is_nez(c)) ) c /= uf;
            /* reduce a0 into [sqrt(2)/2, sqrt(2)] */
            iu     = (iu & 0x007fffff) + 0x3f3504f3;
            T f    = dec(bit_cast(iu, as<T>()));
            T s    = f / (2.0f + f);
            T z    = sqr(s);
            T w    = sqr(z);
            T R    = fma(w,
                         eve::reverse_horner(w, T(0x1.999c26p-2f), T(0x1.f13c4cp-3f))
                        , z * eve::reverse_horner(w, T(0x1.555554p-1f), T(0x1.23d3dcp-2f))
                        );
            T hfsq = half(eve::as<T>()) * sqr(f);
            T dk   = float32(k);
            T r    = fma(dk, Log_2hi, ((fma(s, (hfsq + R), fma(dk, Log_2lo, c)) - hfsq) + f));
            T zz;
            if constexpr( eve::platform::supports_infinites )
            {
              zz = if_else(
                isnez, if_else(a0 == inf(eve::as<T>()), inf(eve::as<T>()), r), minf(eve::as<T>()));
            }
            else { zz = if_else(isnez, r, minf(eve::as<T>())); }
            return if_else(is_ngez(uf), eve::allbits, zz);
          }
          else if constexpr( std::is_same_v<elt_t, double> )
          {
            /* origin: FreeBSD /usr/src/lib/msun/src/e_log1pf.c */
            /*
             * ====================================================
             * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
             *
             * Developed at SunPro, a Sun Microsystems, Inc. business.
             * Permission to use, copy, modify, and distribute this
             * software is freely granted, provided that this notice
             * is preserved.
             * ====================================================
             */
            /* reduce x into [sqrt(2)/2, sqrt(2)] */
            uiT hu = bit_cast(uf, as<uiT>()) >> 32;
            hu += 0x3ff00000 - 0x3fe6a09e;
            iT k = bit_cast(hu >> 20, as<iT>()) - 0x3ff;
            /* correction term ~ log(1+x)-log(u), avoid underflow in c/u */
            T c = if_else(k < 54, if_else(k >= 2, oneminus(uf - a0), a0 - dec(uf)), zero);
            if( eve::any(eve::is_nez(c)) ) c /= uf;
            hu  = (hu & 0x000fffffull) + 0x3fe6a09e;
            T f = bit_cast(bit_cast(hu << 32, as<uiT>()) | ((bit_cast(uf, as<uiT>()) & 0xffffffffull)),
                           as<T>());
            f   = dec(f);

            T hfsq = half(eve::as<T>()) * sqr(f);
            T s    = f / (2.0 + f);
            T z    = sqr(s);
            T w    = sqr(z);
            T t1   = w *
              eve::reverse_horner(w, T(0x1.999999997fa04p-2), T(0x1.c71c51d8e78afp-3), T(0x1.39a09d078c69fp-3))
              ;
            T t2   = z
              *
              eve::reverse_horner(w, T(0x1.5555555555593p-1), T(0x1.2492494229359p-2)
                                 , T(0x1.7466496cb03dep-3), T(0x1.2f112df3e5244p-3))
              ;
            T R  = t2 + t1;
            T dk = float64(k);
            T r  = fma(dk, Log_2hi, ((fma(s, (hfsq + R), fma(dk, Log_2lo, c)) - hfsq) + f));
            T zz;
            if constexpr( eve::platform::supports_infinites )
            {
              zz = if_else(
                isnez, if_else(a0 == inf(eve::as<T>()), inf(eve::as<T>()), r), minf(eve::as<T>()));
            }
            else { zz = if_else(isnez, r, minf(eve::as<T>())); }
            return if_else(is_ngez(uf), eve::allbits, zz);
          }
        }
      }
      else return apply_over(log1p, a0);
    }
    else  // scalar case
    {
      auto x = a0;
      using uiT = as_integer_t<T, unsigned>;
      using iT  = as_integer_t<T, signed>;
      T Log_2hi = ieee_constant<0x1.6300000p-1f, 0x1.62e42fee00000p-1>(eve::as<T>{});
      T Log_2lo = ieee_constant<-0x1.bd01060p-13f, 0x1.a39ef35793c76p-33>(eve::as<T>{});
      if constexpr( std::is_same_v<T, float> )
      {
        /* origin: FreeBSD /usr/src/lib/msun/src/e_log1pf.c */
        /*
         * ====================================================
         * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
         *
         * Developed at SunPro, a Sun Microsystems, Inc. business.
         * Permission to use, copy, modify, and distribute this
         * software is freely granted, provided that this notice
         * is preserved.
         * ====================================================
         */
        uiT ix = bit_cast(x, as<uiT>());
        iT  k  = 1;
        T   c = zero(eve::as<T>()), f = x;
        if( ix < 0x3ed413d0 || ix >> 31 ) /* 1+x < sqrt(2)+  */
        {
          if( ix >= 0xbf800000 ) /* x <= -1.0 */
          {
            if( x == mone(eve::as<T>()) ) return minf(eve::as<T>()); /* log1p(-1)=-inf */
            return nan(eve::as<T>());                                /* log1p(x<-1)=NaN */
          }
          if( ix << 1 < 0x33800000 << 1 ) /* |x| < 2**-24 */
          {
            if( (ix & 0x7f800000) == 0 ) return x;
          }
          if( ix <= 0xbe95f619 ) /* sqrt(2)/2- <= 1+x < sqrt(2)+ */ { k = 0; }
        }
        else if( ix >= 0x7f800000 ) return x;
        if( k )
        {
          /* reduce u into [sqrt(2)/2, sqrt(2)] */
          T   uf = inc(x);
          uiT iu = bit_cast(uf, as<uiT>());
          iu += 0x3f800000 - 0x3f3504f3;
          k = bit_cast(iu >> 23, as<iT>()) - 0x7f;
          /* correction term ~ log(1+x)-log(u), avoid underflow in c/u */
          if( k < 25 )
          {
            c = (k >= 2) ? oneminus(uf - x) : x - dec(uf);
            if( eve::is_nez(c) ) c /= uf;
          }

          /* reduce u into [sqrt(2)/2, sqrt(2)] */
          iu = (iu & 0x007fffff) + 0x3f3504f3;
          f  = dec(bit_cast(iu, as<T>()));
        }
        T s    = f / (2.0f + f);
        T z    = sqr(s);
        T w    = sqr(z);
        T t1   = w *
          eve::reverse_horner(w, T(0x1.999c26p-2f), T(0x1.f13c4cp-3f))
          ;
        T t2   = z *
          eve::reverse_horner(w, T(0x1.555554p-1f), T(0x1.23d3dcp-2f))
          ;
        T R    = t2 + t1;
        T hfsq = 0.5f * sqr(f);
        T dk   = k;
        return fma(dk, Log_2hi, ((fma(s, (hfsq + R), fma(dk, Log_2lo, c)) - hfsq) + f));
      }
      else if constexpr( std::is_same_v<T, double> )
      {
        /* origin: FreeBSD /usr/src/lib/msun/src/e_log1p.c */
        /*
         * ====================================================
         * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
         *
         * Developed at SunSoft, a Sun Microsystems, Inc. business.
         * Permission to use, copy, modify, and distribute this
         * software is freely granted, provided that this notice
         * is preserved.
         * ====================================================
         */
        uiT hx = bit_cast(x, as<uiT>()) >> 32;
        iT  k  = 1;

        T c = zero(eve::as<T>());
        T f = x;
        if( hx < 0x3fda827a || hx >> 31 ) /* 1+x < sqrt(2)+ */
        {
          if( hx >= 0xbff00000 ) /* x <= -1.0 */
          {
            if( x == mone(eve::as<T>()) ) return minf(eve::as<T>()); /* log1p(-1)=-inf */
            return nan(eve::as<T>());                                /* log1p(x<-1)=NaN */
          }
          if( hx << 1 < 0x3ca00000 << 1 ) /* |x| < 2**-53 */
          {
            if( (hx & 0x7ff00000) == 0 ) return x;
          }
          if( hx <= 0xbfd2bec4 ) /* sqrt(2)/2- <= 1+x < sqrt(2)+ */ { k = 0; }
        }
        else if( hx >= 0x7ff00000 ) return x;
        if( k )
        {
          /* reduce x into [sqrt(2)/2, sqrt(2)] */
          T   uf = inc(x);
          uiT hu = bit_cast(uf, as<uiT>()) >> 32;
          hu += 0x3ff00000 - 0x3fe6a09e;
          k = (int)(hu >> 20) - 0x3ff;
          /* correction term ~ log(1+x)-log(u), avoid underflow in c/u */
          if( k < 54 )
          {
            c = (k >= 2) ? oneminus(uf - x) : x - dec(uf);
            if( eve::is_nez(c) ) c /= uf;
          }
          hu = (hu & 0x000fffff) + 0x3fe6a09e;
          f  = bit_cast(bit_cast(hu << 32, as<uiT>()) | ((bit_cast(uf, as<uiT>()) & 0xffffffffull)),
                        as<T>());
          f  = dec(f);
        }

        T hfsq = 0.5 * sqr(f);
        T s    = f / (2.0f + f);
        T z    = sqr(s);
        T w    = sqr(z);
        T t1   = w *
          eve::reverse_horner(w, T(0x1.999999997fa04p-2), T(0x1.c71c51d8e78afp-3), T(0x1.39a09d078c69fp-3))
          ;
        T t2   = z
          *
          eve::reverse_horner(w, T(0x1.5555555555593p-1), T(0x1.2492494229359p-2)
                             , T(0x1.7466496cb03dep-3), T(0x1.2f112df3e5244p-3))
          ;
        T R  = t2 + t1;
        T dk = k;
        return fma(dk, Log_2hi, ((fma(s, (hfsq + R), dk * Log_2lo + c) - hfsq) + f));
      }
    }
  }
}
