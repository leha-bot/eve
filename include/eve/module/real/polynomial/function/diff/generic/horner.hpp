//==================================================================================================
/**
  EVE - Expressive Vector Engine
  Copyright : EVE Contributors & Maintainers
  SPDX-License-Identifier: MIT
**/
//==================================================================================================
#pragma once

#include <eve/detail/overload.hpp>
#include <eve/concept/compatible.hpp>
#include <eve/concept/value.hpp>
#include <eve/constant/zero.hpp>
#include <eve/detail/apply_over.hpp>
#include <eve/detail/implementation.hpp>
#include <eve/detail/skeleton_calls.hpp>
#include <eve/function/derivative.hpp>
#include <eve/function/pedantic/fma.hpp>
#include <eve/function/regular.hpp>
#include <eve/module/real/polynomial/detail/diff_horner_impl.hpp>
#include <eve/function/pedantic.hpp>
#include <eve/function/numeric.hpp>
#include <eve/function/regular.hpp>

namespace eve::detail
{

  //================================================================================================
  //== variadic
  //================================================================================================
  template<int N, value T0, value T1>
  EVE_FORCEINLINE constexpr auto horner_(EVE_SUPPORTS(cpu_)
                                        , diff_type<N> const &
                                        , T0 const &, T1 const &) noexcept
  {
    using r_t = common_compatible_t<T0, T1>;
    return zero(as<r_t>());
  }

  template<int N, value T0, value T1, value T2>
  EVE_FORCEINLINE constexpr auto horner_(EVE_SUPPORTS(cpu_)
                                        , diff_type<N> const &
                                        , T0 const &, T1 const &a, T2 const &) noexcept
  {
    using r_t = common_compatible_t<T0, T1, T2>;
    if constexpr(N == 1) return r_t(a);
    else                 return zero(as<r_t>());
  }

  template<auto N,
           value T0,
           value T1,
           value T2,
           value ...Ts>
           EVE_FORCEINLINE constexpr auto horner_(EVE_SUPPORTS(cpu_)
                                                 , diff_type<N> const &
                                                 , T0 x, T1 a, T2 b, Ts... args) noexcept
  {
    return diff_horner_impl<N>(regular_type(), x, a, b, args...);
  }

  //================================================================================================
  //== variadic with one leading parameter
  //================================================================================================
  template<auto N, value T0>
  EVE_FORCEINLINE constexpr auto horner_(EVE_SUPPORTS(cpu_)
                                        , diff_type<N> const &
                                        , T0 const &
                                        , callable_one_ const &
                                        ) noexcept
  {
    return zero(as<T0>());
  }

  template<auto N, decorator D, value T0, value T1>
  EVE_FORCEINLINE constexpr auto horner_(EVE_SUPPORTS(cpu_)
                                        , diff_type<N> const &
                                        , T0 const &
                                        , callable_one_ const &
                                        , T1 const &) noexcept
  {
    using r_t = common_compatible_t<T0, T1>;
    if constexpr(N == 1) return r_t(1);
    else                 return zero(as<r_t>());
  }

  template<auto N, value T0, value T2, value ...Ts>
  EVE_FORCEINLINE constexpr auto horner_(EVE_SUPPORTS(cpu_)
                                        , diff_type<N> const &
                                        , T0 x
                                        , callable_one_ const &
                                        , T2 b, Ts... args) noexcept
  {
    return diff_horner_impl<N>(regular_type(), x, one, b, args...);
  }

  //================================================================================================
  //== variadic with decorator
  //================================================================================================
  template<auto N, typename D, value T0, value T1>
  EVE_FORCEINLINE constexpr auto horner_(EVE_SUPPORTS(cpu_)
                                        , decorated<diff_<N>(D)> const &
                                        , T0 const &, T1 const &) noexcept
  {
    using r_t = common_compatible_t<T0, T1>;
    return zero(as<r_t>());
  }

  template<auto N, typename D, value T0, value T1, value T2>
  EVE_FORCEINLINE constexpr auto horner_(EVE_SUPPORTS(cpu_)
                                        , decorated<diff_<N>(D)> const &
                                        , T0 const &, T1 const &a, T2 const &) noexcept
  {
    using r_t = common_compatible_t<T0, T1, T2>;
    if constexpr(N == 1) return r_t(a);
    else                 return zero(as<r_t>());
  }

  template<auto N, typename D, value T0, value T1, value T2, value ...Ts>
  EVE_FORCEINLINE constexpr
  auto horner_(EVE_SUPPORTS(cpu_)
              , decorated<diff_<N>(D)> const &
              , T0 x, T1 a, T2 b, Ts... args) noexcept
  {
    return diff_horner_impl<N>(decorated<D()>(), x, a, b, args...);
  }

  //================================================================================================
  //== variadic with decorator and one leading parameter
  //================================================================================================
  template<auto N, typename D, value T0>
  EVE_FORCEINLINE constexpr
  auto horner_(EVE_SUPPORTS(cpu_)
              , decorated<diff_<N>(D)> const &
              , T0 const &
              , callable_one_ const &
              ) noexcept
  {
    return zero(as<T0>());
  }

  template<auto N, typename D, value T0, value T1>
  EVE_FORCEINLINE constexpr
  auto horner_(EVE_SUPPORTS(cpu_)
              , decorated<diff_<N>(D)> const &
              , T0 const &
              , callable_one_ const &
              , T1 const &) noexcept
  {
    using r_t = common_compatible_t<T0, T1>;
    if constexpr(N == 1) return r_t(1);
    else                 return zero(as<r_t>());
  }

  template<auto N, typename D, value T0, value T1, value ...Ts>
  EVE_FORCEINLINE constexpr
  auto horner_(EVE_SUPPORTS(cpu_)
              , decorated<diff_<N>(D)> const &
              , T0 const & x
              , callable_one_ const &
              , T1 b, Ts... args) noexcept
  {
    return diff_horner_impl<N>(decorated<D()>(), x, one, b, args...);
  }

  /////////////////////////////////////////////////////////////////////////
  //== Ranges
  /////////////////////////////////////////////////////////////////////////
  template<value T0, range R>
  EVE_FORCEINLINE constexpr
  auto horner_(EVE_SUPPORTS(cpu_)
              , diff_type<1> const &
              , T0 x, R const &r) noexcept
  requires ((compatible_values<T0, typename R::value_type>) && (!value<R>))
  {
    return diff_horner_impl(eve::regular_type(), x, r);
  }

  template<value T0, typename D, range R>
  EVE_FORCEINLINE constexpr
  auto horner_(EVE_SUPPORTS(cpu_)
              , decorated<diff_<1>(D)> const &
              , T0 x, R const &r) noexcept
  requires ((compatible_values<T0, typename R::value_type>) && (!value<R>))
  {
    return diff_horner_impl(decorated<D()>(), x, r);
  }

  /////////////////////////////////////////////////////////////////////////
  //== Ranges with one leading coef
  /////////////////////////////////////////////////////////////////////////
  template<value T0, range R>
  EVE_FORCEINLINE constexpr
  auto horner_(EVE_SUPPORTS(cpu_)
              , diff_type<1> const &
              , T0 x
              , callable_one_ const &
              , R const &r) noexcept
  requires ((compatible_values<T0, typename R::value_type>) && (!value<R>))
  {
    return diff_horner_impl(regular_type(), x, one, r);
  }

 template<value T0, typename D, range R>
 EVE_FORCEINLINE constexpr auto horner_(EVE_SUPPORTS(cpu_)
                                       , decorated<diff_<1>(D)> const &
                                       , T0 x
                                       , callable_one_ const &
                                       , R const &r) noexcept
  requires ((compatible_values<T0, typename R::value_type>) && (!value<R>))
  {
    return diff_horner_impl(decorated<D()>(), x, one, r);
  }

  /////////////////////////////////////////////////////////////////////////
  //== Iterators
  /////////////////////////////////////////////////////////////////////////
  template<value T0, std::input_iterator IT>
  EVE_FORCEINLINE constexpr
  auto horner_(EVE_SUPPORTS(cpu_)
              , diff_type<1> const &
              , T0 x
              , IT const &first
              , IT const &last) noexcept
  {
    return diff_horner_impl(regular_type(), x, first, last);
  }

  template<value T0, typename  D, std::input_iterator IT>
  EVE_FORCEINLINE constexpr
  auto horner_(EVE_SUPPORTS(cpu_)
              , decorated<diff_<1>(D)> const &
              , T0 x
              , IT const &first
              , IT const &last) noexcept
  {
    return diff_horner_impl(decorated<D()>(), x, first, last);
  }

/////////////////////////////////////////////////////////////////////////
  //== Iterators with leading one
  /////////////////////////////////////////////////////////////////////////
  template<value T0, std::input_iterator IT>
  EVE_FORCEINLINE constexpr
  auto horner_(EVE_SUPPORTS(cpu_)
              , diff_type<1> const &
              , T0 x
              , callable_one_ const &
              , IT const &first
              , IT const &last) noexcept
  {
    return diff_horner_impl(regular_type(), x, one, first, last);
  }

  template<value T0, typename D, std::input_iterator IT>
  EVE_FORCEINLINE constexpr
  auto horner_(EVE_SUPPORTS(cpu_)
              , decorated<diff_<1>(D)> const &
              , T0 x
              , callable_one_ const &
              , IT const &first
              , IT const &last) noexcept
  {
    return diff_horner_impl(decorated<D()>(), x, one, first, last);
  }

}
