//==================================================================================================
/**
  EVE - Expressive Vector Engine
  Copyright 2020 Joel FALCOU
  Copyright 2020 Jean-Thierry LAPRESTE

  Licensed under the MIT License <http://opensource.org/licenses/MIT>.
  SPDX-License-Identifier: MIT
**/
//==================================================================================================
#include <eve/function/is_not_denormal.hpp>
#include <eve/constant/valmin.hpp>
#include <eve/constant/valmax.hpp>
#include <tts/tests/range.hpp>
#include "measures.hpp"
#include "producers.hpp"
#include <type_traits>
#include <cmath>

TTS_CASE("wide exhaustive check on is_not_denormal")
{
  using l_t = eve::as_logical_t<EVE_TYPE>; 
  if constexpr(std::is_floating_point_v<EVE_VALUE>)
  {
    auto std_is_not_denormal = tts::vectorize<l_t>( [](auto e) { return std::fpclassify(e) != FP_SUBNORMAL; } );
    eve::exhaustive_producer<EVE_TYPE> p(eve::Valmin<EVE_VALUE>()+1, eve::Valmax<EVE_VALUE>());
    TTS_RANGE_CHECK(p, std_is_not_denormal, eve::is_not_denormal);
  }
  else
  {
    auto std_is_not_denormal = tts::vectorize<l_t>( [](auto e) { return true; } );
    eve::exhaustive_producer<EVE_TYPE> p(eve::Valmin<EVE_VALUE>(), eve::Valmax<EVE_VALUE>());
    TTS_RANGE_CHECK(p, std_is_not_denormal, eve::is_not_denormal);
  }
  
  
}
