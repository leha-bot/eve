//==================================================================================================
/**
  EVE - Expressive Vector Engine
  Copyright 2020 Joel FALCOU
  Copyright 2020 Jean-Thierry LAPRESTE

  Licensed under the MIT License <http://opensource.org/licenses/MIT>.
  SPDX-License-Identifier: MIT
**/
//==================================================================================================
#include <eve/function/rem_pio2.hpp>

#define TYPE()        double
#define FUNCTION()    eve::rem_pio2
#define SAMPLES(N)    random<T>(N,0.,10000000.)

#include "bench.hpp"
