//==================================================================================================
/*
  EVE - Expressive Vector Engine
  Copyright : EVE Project Contributors
  SPDX-License-Identifier: BSL-1.0
*/
//==================================================================================================
#pragma once

#include <eve/detail/abi.hpp>
#include <eve/forward.hpp>
#include <eve/module/core/regular/store_equivalent.hpp>
#include <eve/module/core/regular/unalign.hpp>
#include <eve/module/core/regular/write.hpp>

namespace eve::detail
{
  template<typename T, typename Idx, typename Ptr, callable_options O>
  EVE_FORCEINLINE void scatter_(EVE_REQUIRES(cpu_), O const& o, T const& v, Ptr p, Idx const& idx)
  {
    // Retrieve element to scatter
    auto se   = store_equivalent(o[condition_key],v,p);

    // Extract the pointer from a potential aligned_ptr
    auto base = unalign(get<2>(se));

    // Single-value scatter
    auto sc = [&](auto n, auto c, auto v)
    {
      // We only write if mask is set
      if constexpr(match_option<condition_key,O,ignore_none_>) write(v.get(n),base+idx.get(n));
      else
      {
        auto m = c.mask( as<as_logical_t<T>>{} );
        if(m.get(n)) write(v.get(n),base+idx.get(n));
      }
    };

    // Scatter all (clang doesn't like capturing structured bindings)
    eve::detail::for_<0, 1, T::size()>([&](auto... I) { ( sc(I,get<0>(se),get<1>(se)), ...); });
  }
}
