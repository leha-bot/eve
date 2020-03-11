//==================================================================================================
/**
  EVE - Expressive Vector Engine
  Copyright 2020 Joel FALCOU
  Copyright 2020 Jean-Thierry LAPRESTE

  Licensed under the MIT License <http://opensource.org/licenses/MIT>.
  SPDX-License-Identifier: MIT
**/
//==================================================================================================
#include <eve/function/extract.hpp>
#include <eve/wide.hpp>

#define TYPE()        eve::wide<int16_t>
#define FUNCTION()    eve::extract
#define SAMPLES(N)    random<int>(N, std::ptrdiff_t(0), TYPE()::static_size-1),random<T>(N,-100.,100.)

#include "bench.hpp"
