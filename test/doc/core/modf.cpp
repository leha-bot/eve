#include <eve/function/modf.hpp>
#include <eve/wide.hpp>
#include <eve/constant/mindenormal.hpp>
#include <eve/constant/minf.hpp>
#include <eve/constant/inf.hpp>
#include <eve/constant/nan.hpp>

using wide_ft = eve::wide<float, eve::fixed<8>>;

int main()
{
  wide_ft pf = {-0.0, 
                1.30f,
                -1.3f,
                eve::Inf<float>(),
                eve::Minf<float>(),
                eve::Nan<float>(),
                0.678f,
                0.0f};

  auto [m, e] = eve::modf(pf); 
  std::cout << "---- simd" << '\n'
            << "<- pf =               " << pf << '\n'
            << "-> eve::modf(pf) =   [" << m << ", \n"
            << "                      " << e << "] \n"; 

  float xf = 2.3;
  auto [sm, se] =  eve::modf(xf); 

  std::cout << "---- scalar" << '\n'
            << "<- xf  =            " << xf << '\n'
            << "-> eve::modf(xf) = [" << sm << ", " << se << "]\n"; 
  return 0;
}
