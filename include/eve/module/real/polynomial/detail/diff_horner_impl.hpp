#pragma once

#include <eve/traits/common_compatible.hpp>
#include <eve/concept/compatible.hpp>
#include <eve/concept/value.hpp>
#include <eve/detail/apply_over.hpp>
#include <eve/detail/concepts.hpp>
#include <eve/detail/implementation.hpp>
#include <eve/detail/skeleton_calls.hpp>
#include <eve/function/add.hpp>
#include <eve/function/pow.hpp>
#include <eve/function/decorator.hpp>
#include <eve/function/fma.hpp>
#include <eve/constant/one.hpp>
#include <eve/concept/range.hpp>
#include <iterator>
#include <initializer_list>

namespace eve::detail
{


  //================================================================================================
  //== N+ 1  parameters (((..(n*a*x+b*n-1)*x+c*n-2)*x + ..)..)
  //================================================================================================
  //==  N = 0
  template<decorator D,  value T0>
  EVE_FORCEINLINE constexpr auto diff_horner_impl(D const &, T0 const &) noexcept
  {
    return T0(0);
  }

  //==  N = 1
  template<decorator D, value T0, value T1>
  EVE_FORCEINLINE constexpr auto diff_horner_impl(D const &
                                            , T0 const &, T1 const &) noexcept
  requires compatible_values<T0, T1>
  {
    using r_t = common_compatible_t<T0, T1>;
    return zero(as<r_t>());
  }

  //==  N = 2
  template<int M, decorator D, value T0, value T1, value T2>
  EVE_FORCEINLINE constexpr auto diff_horner_impl(D const &
                                            , T0 const &x, T1 const &a, T2 const & ) noexcept
  requires compatible_values<T0, T1> && compatible_values<T1, T2>
  {
    using r_t = common_compatible_t<T0, T1>;
    if constexpr(M  == 1)      return r_t(a);
    else if constexpr(M == 2)  return r_t(x);
    else                       return zero(as<r_t>());
  }

  //==  N >= 3
  template<auto M,
           decorator D,
           value T0,
           value T1,
           value T2,
           value ...Ts>
  EVE_FORCEINLINE constexpr
  auto diff_horner_impl(D const & d
                  , T0 xx, T1 a, T2 b, Ts... args) noexcept
  {
    using r_t = common_compatible_t<T0, T1, T2, Ts...>;
    auto x =  r_t(xx);
    if constexpr(M == 1)
    {
      auto dfma = d(fma);
      auto n = sizeof...(args)+1;
      r_t that(dfma(x, n*a, (n-1)*b));
      --n;
      auto next = [x, &n, dfma](auto that, auto arg){
        --n;
        return n ? dfma(that, x, n*arg) : that;
      };
      ((that = next(that, args)),...);
      return that;
    }
    else
    {
      int constexpr N = sizeof...(args)+1;
      if constexpr(M <= N+2) return pow(x, N-M+2);
      else
      {
        using r_t = common_compatible_t<T0, T1>;
        return zero(as<r_t>());
      }
    }
  }

  //================================================================================================
  //== Horner with iterators
  //================================================================================================
  template<decorator D, value T0, std::input_iterator IT>
  EVE_FORCEINLINE constexpr auto diff_horner_impl( D const & d
                                                 , T0 xx
                                                 , IT const & first
                                                 , IT const & last
                                                 ) noexcept
  requires (compatible_values<T0, typename std::iterator_traits<IT>::value_type>)
  {
    using r_t = common_compatible_t<T0, typename std::iterator_traits<IT>::value_type>;
    auto x =  r_t(xx);
    auto n = std::distance(first, last);
    if (n <= 1)
    {
      return r_t(0);
    }
    else    if (n == 2)
    {
      return r_t(*first);
    }
    else
    {
      using std::advance;
      auto cur = first;
      advance(cur, 1);
      auto dfma = d(fma);
      --n;
      r_t that(dfma(x, r_t(*first)*n, r_t(*cur)*r_t(n-1)));
      --n;
      for (advance(cur, 1); n >= 2; advance(cur, 1))
      {
        that = dfma(that, x, r_t(*cur)*r_t(--n));
      }
      return that;
    }
  }

  //================================================================================================
  //== Horner with ranges
  //================================================================================================
  template<decorator D, value T0, range R>
  EVE_FORCEINLINE constexpr auto diff_horner_impl(D const & d
                                        , T0 x, R const & r) noexcept
  requires (compatible_values<T0, typename R::value_type> && (!simd_value<R>))
  {
    return diff_horner_impl(d, x, std::begin(r), std::end(r));
  }

  //================================================================================================
  //== N+ 1  parameters (((..(n*x+'n-&)b)*x+(n-2)*c)*x + ..)..) with unitary leader coefficient
  //================================================================================================
  //==  N = 0,nope one is there
  //==  N = 1
  template<decorator D, value T0>
  EVE_FORCEINLINE constexpr auto diff_horner_impl(D const &
                                            , T0 const & , callable_one_ const &) noexcept
  {
    return zero(as<T0>());
  }
  //==  N = 2
  template<decorator D, value T0, value T2>
  EVE_FORCEINLINE constexpr auto diff_horner_impl(D const &
                                            , T0 const &, callable_one_ const &, T2 const &) noexcept
  requires compatible_values<T0, T2>
  {
    using r_t = common_compatible_t<T0, T2>;
    return one(as<r_t>());
  }

  //==  N >= 3
  template<auto M,
           decorator D,
           value T0,
           value T2,
           value ...Ts>
  EVE_FORCEINLINE constexpr
  auto diff_horner_impl(D const & d
                  , T0 xx, callable_one_ const&, T2 b, Ts... args) noexcept
  {
    using r_t = common_compatible_t<T0, T2, Ts...>;
    auto x =  r_t(xx);
    if constexpr(M == 1)
    {
      auto dfma = d(fma);
      auto n = sizeof...(args)+1;
      r_t that(dfma(x, r_t(n), r_t(n-1)*r_t(b)));
      --n;
      auto next = [x, &n, dfma](auto that, auto arg){
        --n;
        return n >= 1 ? dfma(that, x, r_t(n)*arg) : that;
      };
      ((that = next(that, r_t(args))),...);
      return that;
    }
    else
    {
      int constexpr N = sizeof...(args)+2;
      if constexpr(M <= N+2) return pow(r_t(x), N-M+2);
      else
      {
        return zero(as<r_t>());
      }
    }
  }


  //================================================================================================
  //== Horner with iterators and unitary leader coefficient
  //================================================================================================
  template<decorator D, value T0, std::input_iterator IT>
  EVE_FORCEINLINE constexpr auto diff_horner_impl( D const & d
                                            , T0 xx
                                           ,  callable_one_ const &
                                            , IT const & first
                                            , IT const & last
                                            ) noexcept
  requires (compatible_values<T0, typename std::iterator_traits<IT>::value_type>)
  {
    using r_t = common_compatible_t<T0, typename std::iterator_traits<IT>::value_type>;
    auto x =  r_t(xx);
    auto n = std::distance(first, last)+1;
    if (n <= 1)
    {
      return r_t(0);
    }
    else if (n == 2)
    {
      return r_t(1);
    }
    else
    {
      using std::advance;
      auto cur = first;
      auto dfma = d(fma);
      --n;
      r_t that(dfma(x, n, r_t(*cur)*(n-1)));
      --n;
      for (advance(cur, 1); n >= 2; advance(cur, 1))
      {
        that = dfma(that, x, r_t(*cur)*(--n));
      }
      return that;
    }
  }


  //================================================================================================
  //== Horner with ranges and leading unitary coefficient
  //================================================================================================
  template<decorator D, value T0, range R>
  EVE_FORCEINLINE constexpr auto diff_horner_impl(D const & d
                                                 , T0 x
                                                 , callable_one_ const &
                                                 , R const & r) noexcept
  requires ((compatible_values<T0, typename R::value_type>) && (!simd_value<R>))
  {
  return diff_horner_impl(d, x, one, std::begin(r), std::end(r));
  }

}
