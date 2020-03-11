//==================================================================================================
/**
  EVE - Expressive Vector Engine
  Copyright 2020 Joel FALCOU
  Copyright 2020 Jean-Thierry LAPRESTE

  Licensed under the MIT License <http://opensource.org/licenses/MIT>.
  SPDX-License-Identifier: MIT
**/
//==================================================================================================
#include <eve/function/cosd.hpp>
#include <eve/wide.hpp>
#include <eve/constant/pi.hpp>

#define TYPE()        eve::wide<float>
#define FUNCTION()    eve::cosd
#define SAMPLES(N)    random<T>(N,-1000000*eve::Pi<T>(),1000000*eve::Pi<T>())

#include "bench.hpp"
