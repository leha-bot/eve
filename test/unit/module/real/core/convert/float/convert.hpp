//==================================================================================================
/**
  EVE - Expressive Vector Engine
  Copyright 2020 Joel FALCOU
  Copyright 2020 Jean-Thierry LAPRESTE

  Licensed under the MIT License <http://opensource.org/licenses/MIT>.
  SPDX-License-Identifier: MIT
**/
//==================================================================================================
#include <eve/function/convert.hpp>
#include <eve/constant/valmin.hpp>
#include <eve/constant/valmax.hpp>
#include <eve/function/converter.hpp>
#include <eve/function/sqr.hpp>
#include <type_traits>

TTS_CASE_TPL("Check eve::convert return type", EVE_TYPE)
{
#if defined(EVE_SIMD_TESTS)
  using target_t = eve::wide<float, eve::fixed<EVE_CARDINAL>>;
#else
  using target_t = float;
#endif

  TTS_EXPR_IS(eve::convert(T(), eve::as<float>()), target_t);
  TTS_EXPR_IS(eve::float32(T())     , target_t);
}

TTS_CASE_TPL("Check eve::convert behavior", EVE_TYPE)
{
#if defined(EVE_SIMD_TESTS)
  using target_t = eve::wide<float, eve::fixed<EVE_CARDINAL>>;
#else
  using target_t = float;
#endif

  using v_t = eve::element_type_t<T>;

  TTS_EQUAL(eve::convert((T(0))          , eve::as<float>()), static_cast<target_t>(0) );
  TTS_EQUAL(eve::convert((T(42.69))      , eve::as<float>()), static_cast<target_t>(v_t(42.69)) );

  if constexpr(sizeof(v_t)<=sizeof(float))
  {
    TTS_EQUAL(eve::convert(eve::valmin(eve::as<T>()), eve::as<float>()), static_cast<target_t>(eve::valmin(eve::as<v_t>())) );
    TTS_EQUAL(eve::convert(eve::valmax(eve::as<T>()), eve::as<float>()), static_cast<target_t>(eve::valmax(eve::as<v_t>())) );
  }
}

TTS_CASE_TPL("float32", EVE_TYPE)
{
  TTS_EQUAL(eve::convert((T(0))        , eve::as<float>()), eve::float32(T(0)) );
  TTS_EQUAL(eve::convert((T(42.69))    , eve::as<float>()), eve::float32(T(42.69)));
  TTS_EQUAL(eve::convert(eve::sqr(T(2)), eve::as<float>()), eve::float32(eve::sqr)(T(2)));
}