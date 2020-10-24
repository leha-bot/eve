//==================================================================================================
/**
  EVE - Expressive Vector Engine
  Copyright 2020 Joel FALCOU
  Copyright 2020 Jean-Thierry LAPRESTE

  Licensed under the MIT License <http://opensource.org/licenses/MIT>.
  SPDX-License-Identifier: MIT
**/
//==================================================================================================
#include <eve/function/cosh.hpp>
#include <eve/function/prev.hpp>
#include <eve/function/next.hpp>
#include <eve/function/is_finite.hpp>
#include <eve/constant/nan.hpp>
#include <eve/constant/inf.hpp>
#include <eve/constant/minf.hpp>
#include <eve/platform.hpp>

TTS_CASE_TPL("Check eve::exp properties", EVE_TYPE)
{
  {
    auto reg = eve::cosh;
    using v_t = eve::element_type_t<T>;
    TTS_ULP_EQUAL (reg(eve::prev(eve::range_min<T>(reg))), eve::inf(eve::as<v_t>()), 0.5);
    TTS_ULP_EQUAL (reg(eve::range_min<T>(reg)), std::cosh(eve::range_min<v_t>(reg)), 0.5);
    TTS_ULP_EQUAL (reg(eve::next(eve::range_max<T>(reg))), eve::inf(eve::as<v_t>()), 0.5);
    TTS_ULP_EQUAL (reg(eve::range_max<T>(reg)), std::cosh(eve::range_max<v_t>(reg)), 0.5);

    auto vmax = eve::range_min<T>(reg)*v_t(0.9);
    auto vmin = eve::range_min<T>(reg)*v_t(1.1);
    if(eve::is_finite(reg(vmax)) && !eve::is_finite(reg(vmin)))
    {
      while(true)
      {
        auto v =  eve::average(vmin, vmax);
        if (eve::is_finite(reg(v))) vmax = v;  else vmin = v;
//         std::cout << "vmin " << vmin <<  std::endl;
//         std::cout << "vmax " << vmax <<  std::endl;
//         std::cout << "vmax > vmin   " << (vmax > vmin) <<  std::endl;
        if(vmin >=   eve::prev(vmax))
        {
          std::cout << std::hexfloat << eve::next(v) << " -> " << reg(eve::next(v)) << " -> " << std::defaultfloat << std::setprecision(16) << eve::next(v) << std::endl;
          std::cout << std::hexfloat << v << " -> " << reg(v) << std::endl;
          std::cout << std::hexfloat << eve::prev(v) << " -> " << reg(eve::prev(v)) << " -> " << std::defaultfloat << std::setprecision(16) << eve::next(v) << std::endl;
          break;
        }
      }
    }
    else
      std::cout << "zut" << std::endl;
  }
}

TTS_CASE_TPL("Check eve::cosh return type", EVE_TYPE)
{
  TTS_EXPR_IS(eve::cosh(T(0)), T);
}

TTS_CASE_TPL("Check eve::eve::cosh behavior", EVE_TYPE)
{
  if constexpr( eve::platform::supports_invalids )
  {
    TTS_IEEE_EQUAL(eve::cosh(eve::nan(eve::as<T>())) , (eve::nan(eve::as<T>())) );
    TTS_IEEE_EQUAL(eve::cosh(eve::inf(eve::as<T>())) , (eve::inf(eve::as<T>())) );
    TTS_IEEE_EQUAL(eve::cosh(eve::minf(eve::as<T>())), (eve::inf(eve::as<T>())) );
  }

  TTS_ULP_EQUAL(eve::cosh(T(0.5)), T(std::cosh(0.5)), 0.5);
  TTS_ULP_EQUAL(eve::cosh(T(-0.5)),T(std::cosh(-0.5)), 0.5);
  TTS_ULP_EQUAL(eve::cosh(T(1)), T(std::cosh(1.0)), 0.5);
  TTS_ULP_EQUAL(eve::cosh(T(-1)),T(std::cosh(-1.0)), 0.5);
  TTS_ULP_EQUAL(eve::cosh(T(2)), T(std::cosh(2.0)), 0.5);
  TTS_ULP_EQUAL(eve::cosh(T(-2)),T(std::cosh(-2.0)), 0.5);
}
