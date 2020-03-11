//==================================================================================================
/**
  EVE - Expressive Vector Engine
  Copyright 2020 Joel FALCOU
  Copyright 2020 Jean-Thierry LAPRESTE

  Licensed under the MIT License <http://opensource.org/licenses/MIT>.
  SPDX-License-Identifier: MIT
**/
//==================================================================================================
#ifndef EVE_MODULE_CORE_FUNCTION_SIMD_COMMON_BIT_ANDNOT_HPP_INCLUDED
#define EVE_MODULE_CORE_FUNCTION_SIMD_COMMON_BIT_ANDNOT_HPP_INCLUDED

#include <eve/detail/overload.hpp>
#include <eve/detail/skeleton.hpp>
#include <eve/detail/abi_cast.hpp>
#include <eve/detail/meta.hpp>
#include <eve/detail/abi.hpp>
#include <eve/function/bit_cast.hpp>
#include <eve/concept/vectorizable.hpp>
#include <eve/concept/vectorized.hpp>
#include <eve/forward.hpp>
#include <type_traits>

namespace eve::detail
{
  template<typename T, typename U>
  EVE_FORCEINLINE auto bit_andnot_(EVE_SUPPORTS(cpu_), T const &a, U const &b) noexcept
  requires( std::conditional_t<is_vectorized_v<T>, T, U>,
            bit_compatible<T,U>,
            detail::either<is_vectorized_v<T>, is_vectorized_v<U>>)
  {
    using t_abi = abi_type_t<T>;
    using u_abi = abi_type_t<U>;
    using vt_t  = value_type_t<T>;
    using vu_t  = value_type_t<U>;

    if constexpr(is_vectorizable_v<T> && !is_vectorizable_v<U>)
    {
      if constexpr(sizeof(T) == sizeof(vu_t))
      // this will ensure that no scalar conversion will take place in aggregated
      // in the case vector and scalar not of the value type
      {
        return eve::bit_andnot(U(bit_cast(a,as_<vu_t>())), b);
      }
      else return U();
    }
    else if constexpr(is_vectorizable_v<U> && !is_vectorizable_v<T>)
    {
      if constexpr(sizeof(U) == sizeof(vt_t))
      // this will ensure that no scalar conversion will take place in aggregated
      // in the case vector and scalar not of the value type
      {
        return eve::bit_andnot(a, T(bit_cast(b,as_<vt_t>())));
      }
      else return T();
    }
    else if constexpr(is_emulated_v<t_abi> || is_emulated_v<u_abi>)
    {
      return map(eve::bit_andnot, abi_cast<value_type_t<U>>(a), abi_cast<vt_t>(b));
    }
    else if constexpr(is_aggregated_v<t_abi> || is_aggregated_v<u_abi>)
    {
      return aggregate(eve::bit_andnot, abi_cast<value_type_t<U>>(a), abi_cast<vt_t>(b));
    }
    else if constexpr(is_vectorized_v<T> && is_vectorized_v<U>)
    {
      return eve::bit_andnot(a, bit_cast(b,as(a)));
    }
    else
    {
      static_assert(wrong<T, U>, "[eve::bit_andnot] - Missing implementation");
      return T();
    }
  }
}

#endif
