//==================================================================================================
/*
  EVE - Expressive Vector Engine
  Copyright : EVE Project Contributors
  SPDX-License-Identifier: BSL-1.0
*/
//==================================================================================================
#pragma once

#include <eve/arch.hpp>
#include <eve/traits/overload.hpp>
#include <eve/module/core/decorator/core.hpp>

namespace eve
{
  template<typename Options>
  struct minexponent_t : constant_callable<minexponent_t, Options, downward_option, upward_option>
  {
    template<floating_value T>
    static EVE_FORCEINLINE constexpr auto value(eve::as<T> const&, auto const&)
    {
      using e_t = element_type_t<T>;
      using i_t = as_integer_t<T>;

      if constexpr(std::same_as<e_t, float>  ) return  i_t(-126);
      else if constexpr(std::same_as<e_t, double> ) return  i_t(-1022);
    }

    template<floating_value T>
    requires(plain_scalar_value<element_type_t<T>>)
      EVE_FORCEINLINE constexpr auto operator()(as<T> const& v) const { return EVE_DISPATCH_CALL(v); }

    EVE_CALLABLE_OBJECT(minexponent_t, minexponent_);
  };

//================================================================================================
//! @addtogroup core_constants
//! @{
//!   @var minexponent
//!   @brief Computes the  the greatest exponent of a floating point IEEE value
//!
//!   **Defined in Header**
//!
//!   @code
//!   #include <eve/module/core.hpp>
//!   @endcode
//!
//!   @groupheader{Callable Signatures}
//!
//!   @code
//!   namespace eve
//!   {
//!      template< eve::value T >
//!      eve::as_integer<T> minexponent(as<T> x) noexcept;
//!   }
//!   @endcode
//!
//!   **Parameters**
//!
//!     * `x` :  [Type wrapper](@ref eve::as) instance embedding the type of the constant.
//!
//!    **Return value**
//!
//!      The call `eve::limitexponent(as<T>())` is semantically equivalent to
//!       * `as_integer_t<T>(-126)` if `eve::element_type_t<T>` is float
//!       * `as_integer_t<T>(-1022)` if `eve::element_type_t<T>` is double
//!
//!  @groupheader{Example}
//!
//!  @godbolt{doc/core/constant/minexponent.cpp}
//! @}
//================================================================================================
 inline constexpr auto minexponent = functor<minexponent_t>;
}
