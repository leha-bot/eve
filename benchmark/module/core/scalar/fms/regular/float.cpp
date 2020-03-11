//==================================================================================================
/**
  EVE - Expressive Vector Engine
  Copyright 2020 Joel FALCOU
  Copyright 2020 Jean-Thierry LAPRESTE

  Licensed under the MIT License <http://opensource.org/licenses/MIT>.
  SPDX-License-Identifier: MIT
**/
//==================================================================================================
#include <eve/function/fms.hpp>

#define TYPE()        float
#define FUNCTION()    eve::fms
#define SAMPLES(N)    random<T>(N,-100.f,100.f),random<T>(N,-100.f,100.f),random<T>(N,-100.,100.)

#include "bench.hpp"
