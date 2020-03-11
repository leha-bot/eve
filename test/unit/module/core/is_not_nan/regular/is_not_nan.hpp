//==================================================================================================
/**
  EVE - Expressive Vector Engine
  Copyright 2020 Joel FALCOU
  Copyright 2020 Jean-Thierry LAPRESTE

  Licensed under the MIT License <http://opensource.org/licenses/MIT>.
  SPDX-License-Identifier: MIT
**/
//==================================================================================================
#include <eve/function/is_not_nan.hpp>
#include <eve/constant/false.hpp>
#include <eve/constant/true.hpp>
#include <eve/constant/nan.hpp>
#include <tts/tests/relation.hpp>
#include <tts/tests/types.hpp>
#include <type_traits>

TTS_CASE("Check eve::is_not_nan return type")
{
  TTS_EXPR_IS(eve::is_not_nan(EVE_TYPE(0)), (eve::logical<EVE_TYPE>));
}

TTS_CASE("Check eve::is_not_nan behavior")
{
  TTS_EQUAL(eve::is_not_nan(EVE_TYPE(0)), eve::True<EVE_TYPE>());
  TTS_EQUAL(eve::is_not_nan(EVE_TYPE(2)), eve::True<EVE_TYPE>());

  if constexpr(std::is_floating_point_v<EVE_VALUE> && eve::platform::supports_nans)
  {
    TTS_EQUAL(eve::is_not_nan(eve::Nan<EVE_TYPE>()), eve::False<EVE_TYPE>());
  }
}
