//==================================================================================================
/**
  EVE - Expressive Vector Engine
  Copyright 2019 Joel FALCOU
  Copyright 2019 Jean-Thierry LAPRESTE

  Licensed under the MIT License <http://opensource.org/licenses/MIT>.
  SPDX-License-Identifier: MIT
**/
//==================================================================================================
#ifndef EVE_MODULE_CORE_FUNCTION_SCALAR_MAX_HPP_INCLUDED
#define EVE_MODULE_CORE_FUNCTION_SCALAR_MAX_HPP_INCLUDED

#include <eve/detail/overload.hpp>
#include <eve/detail/abi.hpp>
#include <type_traits>
#include <eve/tags.hpp>
#include <eve/concept/vectorizable.hpp>

namespace eve::detail
{
  // -----------------------------------------------------------------------------------------------
  // Regular case
  template<typename T>
  EVE_FORCEINLINE constexpr auto max_(EVE_SUPPORTS(cpu_)
                                     , T const &a0
                                     , T const &a1) noexcept
  requires(T, Arithmetic<T>)
  {
     return (a0 < a1) ? a1 : a0;
  }
  
  // -----------------------------------------------------------------------------------------------
  // Pedantic case
  template<typename T>
  EVE_FORCEINLINE constexpr auto max_(EVE_SUPPORTS(cpu_)
                                     , pedantic_type const &
                                     , T const &a0
                                     , T const &a1) noexcept
  requires(T, Arithmetic<T>)
  {
    return (a0 < a1) ? a1 : a0;
  }  
    
}

#endif
