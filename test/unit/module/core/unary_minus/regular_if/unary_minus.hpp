//==================================================================================================
/**
  EVE - Expressive Vector Engine
  Copyright 2020 Joel FALCOU
  Copyright 2020 Jean-Thierry LAPRESTE

  Licensed under the MIT License <http://opensource.org/licenses/MIT>.
  SPDX-License-Identifier: MIT
**/
//==================================================================================================
#include <eve/function/unary_minus.hpp>
#include <eve/constant/true.hpp>
#include <eve/constant/false.hpp>
#include <eve/logical.hpp>
#include <tts/tests/relation.hpp>
#include <tts/tests/types.hpp>

TTS_CASE("Check conditional eve::unary_minus return type")
{
  TTS_EXPR_IS(eve::unary_minus[ EVE_VALUE(0)  ](EVE_TYPE(0))              , (EVE_TYPE));
  TTS_EXPR_IS(eve::unary_minus[ (EVE_TYPE(0)) ](EVE_TYPE(0))              , (EVE_TYPE));
  TTS_EXPR_IS(eve::unary_minus[ (eve::logical<EVE_TYPE>(0))](EVE_TYPE(0)) , (EVE_TYPE));
  TTS_EXPR_IS(eve::unary_minus[ eve::logical<EVE_VALUE>(0) ](EVE_TYPE(0)) , (EVE_TYPE));
  TTS_EXPR_IS(eve::unary_minus[ true ](EVE_TYPE(0))                   , (EVE_TYPE));
}

TTS_CASE("Check conditional eve::unary_minus behavior")
{
  EVE_TYPE tv(2);
  auto t = eve::True<EVE_TYPE>();
  auto f = eve::False<EVE_TYPE>();

  TTS_EQUAL(eve::unary_minus[ 1 ](tv)   , (EVE_TYPE(-tv)));
  TTS_EQUAL(eve::unary_minus[ 1.0 ](tv) , (EVE_TYPE(-tv)));
  TTS_EQUAL(eve::unary_minus[ true ](tv), (EVE_TYPE(-tv)));
  TTS_EQUAL(eve::unary_minus[ t ](tv)   , (EVE_TYPE(-tv)));

  TTS_EQUAL(eve::unary_minus[ 0 ](tv)     , tv);
  TTS_EQUAL(eve::unary_minus[ 0.0 ](tv)   , tv);
  TTS_EQUAL(eve::unary_minus[ false ](tv) , tv);
  TTS_EQUAL(eve::unary_minus[ f ](tv)     , tv);

    // Mixed case
  eve::as_logical_t<EVE_TYPE> m;
  std::for_each ( tts::detail::begin(m), tts::detail::end(m)
                , [k = true](auto& e) mutable { e = k; k = !k; }
                );

  TTS_EQUAL(eve::unary_minus[ m ](tv), eve::if_else(m,  (EVE_TYPE(-tv)), tv) );
}
