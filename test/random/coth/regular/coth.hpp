//==================================================================================================
/**
  EVE - Expressive Vector Engine
  Copyright 2020 Joel FALCOU
  Copyright 2020 Jean-Thierry LAPRESTE

  Licensed under the MIT License <http://opensource.org/licenses/MIT>.
  SPDX-License-Identifier: MIT
**/
//==================================================================================================
#include <eve/function/coth.hpp>
#include <eve/function/rec.hpp>
#include <eve/constant/valmin.hpp>
#include <eve/constant/valmax.hpp>
#include <tts/tests/range.hpp>
#include "measures.hpp"
#include "producers.hpp"
#include <cmath>

TTS_CASE("wide random check on coth")
{
  auto std_coth = tts::vectorize<EVE_TYPE>( [](auto e) { return eve;::rec(std::tanh(e)); } );

  eve::rng_producer<EVE_TYPE> p(eve::Valmin<EVE_VALUE>(), EVE_VALUE(-1));
  TTS_RANGE_CHECK(p, std_coth, eve::coth); 
  eve::rng_producer<EVE_TYPE> p(EVE_VALUE(1), eve::Valmax<EVE_VALUE>());
  TTS_RANGE_CHECK(p, std_coth, eve::coth); 
}
