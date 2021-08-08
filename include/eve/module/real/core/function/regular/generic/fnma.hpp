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
#include <eve/detail/implementation.hpp>
#include <eve/detail/skeleton_calls.hpp>
#include <eve/function/fma.hpp>

namespace eve::detail
{
  template<real_value T, real_value U, real_value V>
  EVE_FORCEINLINE auto fnma_(EVE_SUPPORTS(cpu_), T const &a, U const &b, V const &c) noexcept
  requires properly_convertible<U, V, T>
  {
    using r_t =  common_compatible_t<T, U, V>;
    return arithmetic_call(fnma, r_t(a), r_t(b), r_t(c));
  }

  template<real_scalar_value T>
  EVE_FORCEINLINE T fnma_(EVE_SUPPORTS(cpu_), T const &a, T const &b, T const &c) noexcept
  {
    return -a * b + c;
  }

  template<real_simd_value T>
  EVE_FORCEINLINE T fnma_(EVE_SUPPORTS(cpu_), T const &a, T const &b, T const &c) noexcept
  requires has_native_abi_v<T>
  {
    return fma(T(-a), b, c);
  }

  //================================================================================================
  // Masked case
  //================================================================================================
  template<conditional_expr C, real_value T, real_value U, real_value V>
  EVE_FORCEINLINE auto fnma_(EVE_SUPPORTS(cpu_), C const &cond, T const &a, U const &b, V const &c) noexcept
  requires properly_convertible<U, V, T>
  {
    using r_t =  common_compatible_t<T, U, V>;
    return mask_op(  cond, eve::fnma, r_t(a), r_t(b), r_t(c));
  }

  //================================================================================================
  // Rounded case
  //================================================================================================
  template<decorator D, real_value T>
  EVE_FORCEINLINE T fnma_(EVE_SUPPORTS(cpu_), D const &, T a, T b, T c) noexcept
  requires  has_native_abi_v<T>
  && (is_one_of<D>(types<toward_zero_type, downward_type, to_nearest_type, upward_type> {}))
  {
    return D()(round)(fnma(a, b, c));
  }

  //================================================================================================
  // Rounded masked case
  //================================================================================================
  template<conditional_expr C, decorator D, real_value T>
  EVE_FORCEINLINE T fnma_(EVE_SUPPORTS(cpu_), C const &cond, D const &, T a, T b, T c) noexcept
  requires  has_native_abi_v<T>
  && (is_one_of<D>(types<toward_zero_type, downward_type, to_nearest_type, upward_type> {}))
  {
    auto tmp = mask_op( cond, eve::fnma, a, b, c);
    return mask_op( cond, D()(eve::round), tmp);
  }
}
