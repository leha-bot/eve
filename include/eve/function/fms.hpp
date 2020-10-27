//==================================================================================================
/**
  EVE - Expressive Vector Engine
  Copyright 2020 Joel FALCOU
  Copyright 2020 Jean-Thierry LAPRESTE

  Licensed under the MIT License <http://opensource.org/licenses/MIT>.
  SPDX-License-Identifier: MIT
**/
//==================================================================================================
#pragma once

#include <eve/detail/overload.hpp>

namespace eve
{
  EVE_MAKE_CALLABLE(fms_, fms);
  namespace detail
  {
    template<> inline constexpr auto supports_pedantic<tag::fms_> = true;
  }
}

#include <eve/arch.hpp>
#include <eve/module/core/function/generic/fms.hpp>

#if defined(EVE_HW_X86)
#  include <eve/module/core/function/simd/x86/fms.hpp>
#endif

#if defined(EVE_HW_ARM)
#  include <eve/module/core/function/simd/arm/neon/fms.hpp>
#endif
