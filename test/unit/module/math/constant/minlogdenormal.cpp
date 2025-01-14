//==================================================================================================
/**
  EVE - Expressive Vector Engine
  Copyright : EVE Project Contributors
  SPDX-License-Identifier: BSL-1.0
**/
//==================================================================================================
#include "test.hpp"

#include <eve/module/core.hpp>
#include <eve/module/math.hpp>

#include <cmath>
#include <iomanip>

//==================================================================================================
// Types tests
//==================================================================================================
TTS_CASE_TPL("Check return types of minlogdenormal", eve::test::simd::ieee_reals)
<typename T>(tts::type<T>)
{
  using v_t = eve::element_type_t<T>;
  using eve::as;

  TTS_EXPR_IS(eve::minlogdenormal(as<T>()), T);
  TTS_EXPR_IS(eve::minlogdenormal(as<v_t>()), v_t);
};
//==================================================================================================
// minlogdenormal  tests on scalar on scalar
//==================================================================================================
TTS_CASE_TPL("Check behavior of minlogdenormal on scalar", eve::test::scalar::ieee_reals)
<typename T>(tts::type<T>)
{
  using eve::as;
  TTS_IEEE_EQUAL(eve::exp(eve::minlogdenormal(as<T>())), eve::zero(as<T>()));
};


//==================================================================================================
// minlogdenormal  tests on wide on scalar
//==================================================================================================
TTS_CASE_TPL("Check behavior of minlogdenormal on scalar", eve::test::simd::ieee_reals)
<typename T>(tts::type<T>)
{
  using eve::as;
  TTS_IEEE_EQUAL(eve::exp(eve::minlogdenormal(as<T>())), eve::zero(as<T>()));
  if constexpr( eve::platform::supports_denormals )
    TTS_EXPECT(eve::all(eve::is_gtz(eve::pedantic(eve::exp)(eve::next(eve::minlogdenormal(as<T>()))))));
};


//==================================================================================================
// simd Tests for masked minlogdenormal
//==================================================================================================
TTS_CASE_WITH("Check behavior of minlogdenormal[mask] on :wide)",
              eve::test::simd::ieee_reals,
              tts::generate(tts::randoms(eve::valmin, eve::valmax),
              tts::logicals(0, 3)))
<typename T, typename M>(T const& a0,
                         M const& mask)
{
  TTS_IEEE_EQUAL(eve::minlogdenormal[mask](eve::as(a0)), eve::if_else(mask, eve::minlogdenormal(eve::as(a0)), eve::zero));
};
