//==================================================================================================
/**
  EVE - Expressive Vector Engine
  Copyright 2020 Joel FALCOU
  Copyright 2020 Jean-Thierry LAPRESTE

  Licensed under the MIT License <http://opensource.org/licenses/MIT>.
  SPDX-License-Identifier: MIT
**/
//==================================================================================================
#include <eve/function/is_ngez.hpp>
#include <eve/constant/false.hpp>
#include <eve/constant/true.hpp>
#include <eve/constant/nan.hpp>
#include <tts/tests/relation.hpp>
#include <tts/tests/types.hpp>

TTS_CASE("Check eve::is_ngez return type")
{
  using eve::logical;

  TTS_EXPR_IS(eve::is_ngez(EVE_TYPE() ), (logical<EVE_TYPE>));
}

TTS_CASE("Check eve::is_ngez behavior")
{
  if constexpr(std::is_signed_v<EVE_VALUE>)
  {
    TTS_EQUAL(eve::is_ngez(EVE_TYPE(-1)), eve::True<EVE_TYPE>());
  }

  if constexpr(eve::platform::supports_nans && std::is_floating_point_v<EVE_VALUE>)
  {
    TTS_EQUAL(eve::is_ngez(eve::Nan<EVE_TYPE>()), eve::True<EVE_TYPE>());
  }

  TTS_EQUAL(eve::is_ngez(EVE_TYPE(0)), eve::False<EVE_TYPE>());
  TTS_EQUAL(eve::is_ngez(EVE_TYPE(3)), eve::False<EVE_TYPE>());
}
