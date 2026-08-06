#pragma once
// Turn a runtime integer in a small known range into a compile-time template
// argument, so a kernel (or any template) can be specialised on it.
//
// The motivating case is CUDA kernels whose loop trip count and shift amounts
// come from a lookup table but are constant for the lifetime of a run: making
// them template parameters lets `#pragma unroll` fully unroll the loop and lets
// the compiler constant-fold the address arithmetic.
//
//   dispatch_const<1, 10>(trip_count, [&](auto n)
//   {
//       my_kernel<n.value><<<grid, block>>>(...);
//   });
//
// The lambda is called with a std::integral_constant<int, V>, so `n.value` is a
// constant expression.  Dispatch is O(1): a static table of thunks indexed by
// `value - LO`, no if-chain and no recursion.  Out-of-range values throw.

#include <array>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

// One thunk per constant in [LO, LO + sizeof...(I)); each calls f with the value
// as an integral_constant.  Non-capturing lambdas, so they decay to plain
// function pointers and the whole table is a flat array.
template <int LO, typename F, std::size_t... I>
static auto make_const_table(std::index_sequence<I...>)
{
    return std::array<void (*)(const F &), sizeof...(I)>{
        {+[](const F &f)
         { f(std::integral_constant<int, LO + (int)I>{}); }...}};
}

template <int LO, int HI, typename F>
static void dispatch_const(int value, const F &f)
{
    static_assert(LO <= HI, "empty dispatch range");
    constexpr int N = HI - LO + 1;
    static const auto table = make_const_table<LO, F>(std::make_index_sequence<N>{});
    if (value < LO || value > HI)
        throw std::runtime_error("dispatch_const: value out of range [" +
                                 std::to_string(LO) + ", " + std::to_string(HI) +
                                 "]: " + std::to_string(value));
    table[value - LO](f);
}
