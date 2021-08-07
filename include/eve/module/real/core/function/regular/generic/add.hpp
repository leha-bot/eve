//==================================================================================================
/**
  EVE - Expressive Vector Engine
  Copyright : EVE Contributors & Maintainers
  SPDX-License-Identifier: MIT
**/
//==================================================================================================
#pragma once

#include <eve/concept/compatible.hpp>
#include <eve/concept/value.hpp>
#include <eve/detail/function/conditional.hpp>
#include <eve/function/round.hpp>
#include <eve/traits/common_compatible.hpp>

namespace eve::detail
{
  //================================================================================================
  // Masked case
  //================================================================================================
  template<conditional_expr C, real_value U, real_value V>
  EVE_FORCEINLINE auto add_(EVE_SUPPORTS(cpu_), C const &cond, U const &t, V const &f) noexcept
      requires compatible_values<U, V>
  {
    return mask_op(  cond, eve::add, t, f);
  }

  //================================================================================================
  //N parameters
  //================================================================================================
  template<decorator D, real_value T0, real_value ...Ts>
  auto add_(EVE_SUPPORTS(cpu_), D const &, T0 a0, Ts... args)
    requires (compatible_values<T0, Ts> && ...)
  {
    common_compatible_t<T0,Ts...> that(a0);
    ((that = D()(add)(that,args)),...);
    return that;
  }

  template<real_value T0, real_value ...Ts>
  auto add_(EVE_SUPPORTS(cpu_), T0 a0, Ts... args)
    requires (compatible_values<T0, Ts> && ...)
  {
    common_compatible_t<T0,Ts...> that(a0);
    ((that = add(that,args)),...);
    return that;
  }

  //================================================================================================
  // Rounded case
  //================================================================================================
  template<decorator D, real_value T>
  EVE_FORCEINLINE T add_(EVE_SUPPORTS(cpu_), D const &, T a, T b) noexcept
  requires  has_native_abi_v<T>
  && (is_one_of<D>(types<toward_zero_type, downward_type, to_nearest_type, upward_type> {}))
  {
    return D()(round)(add(a, b));
  }

  //================================================================================================
  // Rounded masked case
  //================================================================================================
  template<conditional_expr C, decorator D, real_value T>
  EVE_FORCEINLINE T add_(EVE_SUPPORTS(cpu_), C const &cond, D const &, T a, T b) noexcept
  requires  has_native_abi_v<T>
  && (is_one_of<D>(types<toward_zero_type, downward_type, to_nearest_type, upward_type> {}))
  {
    auto tmp = mask_op( cond, eve::add, a, b);
    return mask_op( cond, D()(eve::round), tmp);
  }
}
