//==================================================================================================
/*
  EVE - Expressive Vector Engine
  Copyright : EVE Project Contributors
  SPDX-License-Identifier: BSL-1.0
*/
//==================================================================================================
#pragma once

#include <eve/detail/overload.hpp>

namespace eve
{
//================================================================================================
//! @addtogroup math_hyper
//! @{
//! @var tanh
//!
//! @brief Callable object computing \f$\frac{e^x-e^{-x}}{e^x+e^{-x}}\f$.
//!
//!   **Defined in Header**
//!
//!   @code
//!   #include <eve/module/math.hpp>
//!   @endcode
//!
//!   @groupheader{Callable Signatures}
//!
//!   @code
//!   namespace eve
//!   {
//!      template< eve::floating_value T >
//!      T tanh(T x) noexcept;
//!   }
//!   @endcode
//!
//! **Parameters**
//!
//!   *  `x`:   [floating real value](@ref eve::floating_ordered_value).
//!
//! **Return value**
//!
//!   *  Returns the [elementwise](@ref glossary_elementwise) hyperbolic tangent of the input.
//!
//!      In particular:
//!
//!      * If the element is \f$\pm0\f$, \f$\pm0\f$ is returned.
//!      * If the element is \f$\pm\infty\f$, \f$\pm1\f$ returned.
//!      * If the element is a `NaN`, `NaN` is returned.
//!
//!      * for every z: `eve::tanh(eve::conj(z)) == eve::conj(std::tanh(z))`
//!      * for every z: `eve::tanh(-z)           == -eve::tanh(z)`
//!      * If z is \f$+0\f$, the result is \f$+0\f$
//!      * If z is \f$x+i \infty\f$ (for any non zero finite x), the result is \f$NaN+i NaN\f$
//!      * If z is \f$i \infty\f$  the result is \f$i NaN\f$
//!      * If z is \f$x,NaN\f$ (for any non zero finite x), the result is \f$NaN+i NaN\f$
//!      * If z is \f$i NaN\f$  the result is \f$i NaN\f$
//!      * If z is \f$+\infty,y\f$ (for any finite positive y), the result is \f$1\f$
//!      * If z is \f$+\infty+i \infty\f$, the result is \f$1,\pm 0\f$ (the sign of the imaginary part is unspecified)
//!      * If z is \f$+\infty+i NaN\f$, the result is \f$1\f$ (the sign of the imaginary part is unspecified)
//!      * If z is \f$NaN\f$, the result is \f$NaN\f$
//!      * If z is \f$NaN+i y\f$ (for any non-zero y), the result is \f$NaN+i NaN\f$
//!      * If z is \f$NaN+i NaN\f$, the result is \f$NaN+i NaN\f$
//!
//!  @groupheader{Example}
//!
//!  @godbolt{doc/math/regular/tanh.cpp}
//!
//!  @}
//================================================================================================
EVE_MAKE_CALLABLE(tanh_, tanh);
}

#include <eve/module/math/regular/impl/tanh.hpp>
