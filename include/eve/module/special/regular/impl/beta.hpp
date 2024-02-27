//==================================================================================================
/*
  EVE - Expressive Vector Engine
  Copyright : EVE Project Contributors
  SPDX-License-Identifier: BSL-1.0
*/
//==================================================================================================
#pragma once

#include <eve/module/math.hpp>
#include <eve/module/special/regular/log_abs_gamma.hpp>
#include <eve/module/special/regular/signgam.hpp>
#include <eve/traits/common_value.hpp>

namespace eve::detail
{
  template< typename T0, typename T1, callable_options O>
  constexpr EVE_FORCEINLINE
  eve::common_value_t<T0, T1> beta_(EVE_REQUIRES(cpu_), O const&, T0 const& a0,  T1 const & a1)
  {
    if constexpr(std::same_as<T0, T1>)
    {
      auto y    = a0 + a1;
      auto sign = eve::signgam(a0) * eve::signgam(a1) * eve::signgam(y);
      return sign * exp(log_abs_gamma(a0) + log_abs_gamma(a1) - log_abs_gamma(y));
    }
    else
    {
      return arithmetic_call(beta, a0, a1);
    }
  }
}
