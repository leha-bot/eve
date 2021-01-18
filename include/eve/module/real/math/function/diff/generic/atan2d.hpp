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

#include <eve/function/diff/atan2.hpp>
#include <eve/function/derivative.hpp>
#include <eve/function/radindeg.hpp>

namespace eve::detail
{

  template<floating_real_value T, auto N>
  EVE_FORCEINLINE constexpr T atan2d_(EVE_SUPPORTS(cpu_)
                                   , diff_type<N> const &
                                   , T const &x
                                   , T const &y) noexcept
  {
    return radindeg(diff_type<N>()(atan2)(x, y));
  }

}