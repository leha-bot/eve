//==================================================================================================
/**
  EVE - Expressive Vector Engine
  Copyright 2020 Joel FALCOU
  Copyright 2020 Jean-Thierry LAPRESTE

  Licensed under the MIT License <http://opensource.org/licenses/MIT>.
  SPDX-License-Identifier: MIT
**/
//==================================================================================================
#include <eve/function/sub.hpp>
#include <eve/constant/valmin.hpp>
#include <eve/constant/valmax.hpp>
#include <eve/constant/true.hpp>
#include <eve/constant/false.hpp>
#include <eve/logical.hpp>
#include <tts/tests/relation.hpp>
#include <tts/tests/types.hpp>
#include <type_traits>

TTS_CASE("Check sub return type")
{
  TTS_EXPR_IS( (eve::saturated_(eve::sub[ EVE_TYPE()              ])(EVE_TYPE(), EVE_TYPE())), (EVE_TYPE));
  TTS_EXPR_IS( (eve::saturated_(eve::sub[ eve::logical<EVE_TYPE>()])(EVE_TYPE(), EVE_TYPE())), (EVE_TYPE));
  TTS_EXPR_IS( (eve::saturated_(eve::sub[ true                ])(EVE_TYPE(), EVE_TYPE())), (EVE_TYPE));
}

TTS_CASE("Check saturated conditional sub behavior")
{
  EVE_TYPE tv{eve::Valmax<EVE_TYPE>()};
  EVE_TYPE fv{3};
  auto t = eve::True<EVE_TYPE>();
  auto f = eve::False<EVE_TYPE>();

  TTS_EQUAL(eve::saturated_(eve::sub[ 1 ])(tv, fv)    , eve::saturated_(eve::sub)(tv,fv));
  TTS_EQUAL(eve::saturated_(eve::sub[ 1.0 ])(tv, fv)  , eve::saturated_(eve::sub)(tv,fv));
  TTS_EQUAL(eve::saturated_(eve::sub[ true ])(tv, fv) , eve::saturated_(eve::sub)(tv,fv));
  TTS_EQUAL(eve::saturated_(eve::sub[ t ])(tv, fv)    , eve::saturated_(eve::sub)(tv,fv));

  TTS_EQUAL(eve::saturated_(eve::sub[ 0 ])(tv, fv)    , tv);
  TTS_EQUAL(eve::saturated_(eve::sub[ 0.0 ])(tv, fv)  , tv);
  TTS_EQUAL(eve::saturated_(eve::sub[ false ])(tv, fv), tv);
  TTS_EQUAL(eve::saturated_(eve::sub[ f ])(tv, fv)    , tv);

  // Mixed case
  eve::as_logical_t<EVE_TYPE> m;
  std::for_each ( tts::detail::begin(m), tts::detail::end(m)
                , [k = true](auto& e) mutable { e = k; k = !k; }
                );

  TTS_EQUAL ( eve::saturated_(eve::sub[ m ])(tv, fv),
                  eve::if_else(m, eve::saturated_(eve::sub)(tv,fv), tv)
                );

}
