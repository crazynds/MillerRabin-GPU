/* ─────────────────────────────────────────────────────────────────────────────
 * FILE   src/util/const_dispatch.h
 * ROLE   runtime integer to compile-time template argument
 *
 * HOW    Expands a small known range into a switch that calls the given
 *        lambda with an integral_constant, so a template can be specialised
 *        on a value that is only known at runtime.
 *
 * NOTE   The motivating case is the NTT kernels: their trip counts and
 *        shift amounts come from a launch table but are fixed for the run.
 *        As template parameters they become immediates, which lets #pragma
 *        unroll actually unroll.
 * ───────────────────────────────────────────────────────────────────────────── */
#pragma once

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
