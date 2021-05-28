//==================================================================================================
/**
  EVE - Expressive Vector Engine
  Copyright : EVE Contributors & Maintainers
  SPDX-License-Identifier: MIT
**/
//==================================================================================================
#include "test.hpp"
#include <eve/concept/value.hpp>
#include <eve/constant/valmin.hpp>
#include <eve/constant/valmax.hpp>
#include <eve/constant/nan.hpp>
#include <eve/constant/zero.hpp>
#include <eve/constant/one.hpp>
#include <eve/function/horner.hpp>
#include <eve/function/numeric/horner.hpp>
#include <eve/function/pedantic/horner.hpp>
#include <eve/function/diff/horner.hpp>
#include <eve/logical.hpp>
#include <type_traits>
#include <array>
#include <vector>

//==================================================================================================
//== Types tests
//==================================================================================================
EVE_TEST_TYPES( "Check return types of dif(horner) on wide"
        , eve::test::simd::ieee_reals
        )
<typename T>(eve::as_<T>)
{
 using v_t = eve::element_type_t<T>;
 using i_t = eve::as_integer_t<v_t>;
  using eve::horner;
  using eve::diff;
  using eve::pedantic;
  TTS_EXPR_IS( diff(horner)(T(), T())  , T);
  TTS_EXPR_IS( diff(horner)(T(), T(), T())  , T);
  TTS_EXPR_IS( diff(horner)(T(), v_t(), v_t())  , T);
  TTS_EXPR_IS( diff(horner)(T(), v_t(), int())  , T);
  TTS_EXPR_IS( diff(horner)(T(), v_t(), i_t())  , T);
  TTS_EXPR_IS( diff(horner)(v_t(), v_t(), v_t()), v_t);
  TTS_EXPR_IS( diff(horner)(T(), eve::one, T())  , T);
  TTS_EXPR_IS( diff(horner)(T(), eve::one, T(), T())  , T);
  TTS_EXPR_IS( diff(horner)(T(), eve::one, v_t(), v_t())  , T);
  TTS_EXPR_IS( diff(horner)(T(), eve::one, v_t(), int())  , T);
  TTS_EXPR_IS( diff(horner)(T(), eve::one, v_t(), i_t())  , T);
  TTS_EXPR_IS( diff(horner)(v_t(), eve::one, v_t(), v_t()), v_t);
  TTS_EXPR_IS( diff(horner)(T(), (std::array<v_t, 4>())), T());
  TTS_EXPR_IS( diff(horner)(T(), (std::array<T, 4>())), T());

//    TTS_EXPR_IS( pedantic(diff)(horner)(T(), T())  , T);
//   TTS_EXPR_IS( pedantic(diff)(horner)(T(), T(), T())  , T);
//   TTS_EXPR_IS( pedantic(diff)(horner)(T(), v_t(), v_t())  , T);
//   TTS_EXPR_IS( pedantic(diff)(horner)(T(), v_t(), int())  , T);
//   TTS_EXPR_IS( pedantic(diff)(horner)(T(), v_t(), i_t())  , T);
//   TTS_EXPR_IS( pedantic(diff)(horner)(v_t(), v_t(), v_t()), v_t);
//   TTS_EXPR_IS( pedantic(diff)(horner)(T(), eve::one, T())  , T);
//   TTS_EXPR_IS( pedantic(diff)(horner)(T(), eve::one, T(), T())  , T);
//   TTS_EXPR_IS( pedantic(diff)(horner)(T(), eve::one, v_t(), v_t())  , T);
//   TTS_EXPR_IS( pedantic(diff)(horner)(T(), eve::one, v_t(), int())  , T);
//   TTS_EXPR_IS( pedantic(diff)(horner)(T(), eve::one, v_t(), i_t())  , T);
//   TTS_EXPR_IS( pedantic(diff)(horner)(v_t(), eve::one, v_t(), v_t()), v_t);
//    TTS_EXPR_IS( pedantic(diff)(horner)(T(), (std::array<v_t, 4>())), T());
};

//============================================================================
//== diff(horner) tests
//============================================================================
EVE_TEST( "Check behavior of diff(horner) on wide"
        , eve::test::simd::all_types
        , eve::test::generate(eve::test::ramp(0))
        )
  <typename T>(T const& a0)
{
  using eve::horner;
  using eve::pedantic;
  using eve::fma;
  using eve::diff;
  using eve::one;
  using v_t = eve::element_type_t<T>;

  //============================================================================
  //== variadic
  //============================================================================

  TTS_EQUAL(diff(horner)(a0, T(0)), T(0));
  TTS_EQUAL(diff(horner)(a0, T(1)), T(0));
  TTS_EQUAL(diff(horner)(a0, T(1), T(2)), T(1));
  TTS_EQUAL(diff(horner)(a0, T(1), T(2), T(3)), 2*a0+2);
  TTS_EQUAL(diff(horner)(a0, T(1), T(2), T(3), T(4)), (3*a0*a0+4*a0+3));
  TTS_EQUAL(eve::diff_2nd(horner)(a0, T(1), T(2), T(3), T(4)), a0*a0*a0);
  TTS_EQUAL(eve::diff_3rd(horner)(a0, T(1), T(2), T(3), T(4)), a0*a0);
  TTS_EQUAL(eve::diff_nth<4>(horner)(a0, T(1), T(2), T(3), T(4)), a0);
  TTS_EQUAL(eve::diff_nth<5>(horner)(a0, T(1), T(2), T(3), T(4)), T(1));

  //============================================================================
  //== variadic with leading one
  //============================================================================

  TTS_EQUAL(diff(horner)(a0, one), T(0));
  TTS_EQUAL(diff(horner)(a0, one, T(2)), T(1));
  TTS_EQUAL(diff(horner)(a0, one, T(2), T(3)), 2*a0+2);

  {
    //============================================================================
    //== ranges
    //============================================================================
    std::vector<v_t> tab0; // std does not want array of size 0
    std::array<v_t, 1> tab1 = {1};
    std::array<v_t, 2> tab2 = {1, 2};
    std::array<v_t, 3> tab3 = {1, 2, 3};
    std::array<v_t, 4> tab4 = {1, 2, 3, 4};

    TTS_EQUAL(diff(horner)(a0, tab0), T(0));
    TTS_EQUAL(diff(horner)(a0, tab1), T(0));
    TTS_EQUAL(diff(horner)(a0, tab2), T(1));
    TTS_EQUAL(diff(horner)(a0, tab3), 2*a0+2);
    TTS_EQUAL(diff(horner)(a0, tab4), (3*a0*a0+4*a0+3));
  }
  {
    //============================================================================
    //== ranges + leading coefficient one
    //============================================================================
    std::vector<v_t> tab1 = {};// std does not want array of size 0
    std::array<v_t, 1> tab2 = {2};
    std::array<v_t, 2> tab3 = {2, 3};
    std::array<v_t, 3> tab4 = {2, 3, 4};

   TTS_EQUAL(diff(horner)(a0, one, tab1), T(0));
    TTS_EQUAL(diff(horner)(a0, one, tab2), T(1));
    TTS_EQUAL(diff(horner)(a0, one, tab3), 2*a0+2);
    TTS_EQUAL(diff(horner)(a0, one, tab4), (3*a0*a0+4*a0+3));

  }
  {
    //============================================================================
    //== iterators
    //============================================================================
    std::vector<v_t> tab0; // std does not want array of size 0
    std::array<v_t, 1> tab1 = {1};
    std::array<v_t, 2> tab2 = {1, 2};
    std::array<v_t, 3> tab3 = {1, 2, 3};
    std::array<v_t, 4> tab4 = {1, 2, 3, 4};

    TTS_EQUAL(diff(horner)(a0, &tab0[0], &tab0[0]), T(0));
    TTS_EQUAL(diff(horner)(a0, &tab1[0], &tab1[1]), T(0));
    TTS_EQUAL(diff(horner)(a0, &tab2[0], &tab2[2]), T(1));
    TTS_EQUAL(diff(horner)(a0, tab3.begin(), tab3.end()), 2*a0+2);
    TTS_EQUAL(diff(horner)(a0, tab4.begin(), tab4.end()), (3*a0*a0+4*a0+3));
  }
  {
    //============================================================================
    //== iterators with leading one
    //============================================================================
    std::vector<v_t> tab1 = {};// std does not want array of size 0
    std::array<v_t, 1> tab2 = {2};
    std::array<v_t, 2> tab3 = {2, 3};
    std::array<v_t, 3> tab4 = {2, 3, 4};

    TTS_EQUAL(diff(horner)(a0, one, &tab1[0], &tab1[0]), T(0));
    TTS_EQUAL(diff(horner)(a0, one, &tab2[0], &tab2[1]), T(1));
    TTS_EQUAL(diff(horner)(a0, one, tab3.begin(), tab3.end()), 2*a0+2);
    TTS_EQUAL(diff(horner)(a0, one, tab4), (3*a0*a0+4*a0+3));
  }

};
