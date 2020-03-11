//==================================================================================================
/**
  EVE - Expressive Vector Engine
  Copyright 2020 Joel FALCOU
  Copyright 2020 Jean-Thierry LAPRESTE

  Licensed under the MIT License <http://opensource.org/licenses/MIT>.
  SPDX-License-Identifier: MIT
**/
//==================================================================================================
#include <eve/function/is_even.hpp>

#define TYPE()        int16_t
#define FUNCTION()    eve::is_even
#define SAMPLES(N)    random<T>(N,-100.,100.)

#include "bench.hpp"
