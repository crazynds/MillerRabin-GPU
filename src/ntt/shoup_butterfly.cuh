/* ─────────────────────────────────────────────────────────────────────────────
 * FILE   src/ntt/shoup_butterfly.cuh
 * ROLE   NTT butterflies with precomputed-quotient (Shoup) multiplication
 *
 * HOW    The twiddle w is known ahead of time, so precompute w_q =
 *        floor(w*2^64/p). Then q = umul64hi(w_q, x), r = w*x - q*p taken
 *        mod 2^64 lands in [0, 2p), and one conditional subtract finishes
 *        it. Three 64-bit multiplies instead of the three 128-bit ones a
 *        generic Barrett reduction needs.
 *
 * NOTE   Valid while p < 2^63 and both operands are below p; the table
 *        builder checks the first and the transform maintains the second.
 *        Under MR_NTT_LAZY the second no longer holds — inputs reach 4p — so
 *        the bound was re-verified: for the merge prime (~2^59, hence
 *        4p ~ 2^61) the product stays in [0,2p) and stays congruent for every
 *        x in [0,4p). That check plus the butterfly range algebra were
 *        validated against a reference NTT before any kernel was written; the
 *        margin at the top is real but thin (observed max 3.87p against the
 *        4p ceiling), and it is structural: u < 2p after its subtract and
 *        t < 2p, so u + t < 4p always. Raising the prime past 2^62 breaks it.
 *        Profiling put the butterfly kernels at 47-75% of peak bandwidth
 *        while trivially-structured kernels reach 86-92%, with direct
 *        evidence they were not bandwidth-bound — which is why the lever
 *        here is arithmetic, not memory.
 *
 * CHANGELOG
 *   2026-08-12  Sparse q*p behind MR_NTT_SPARSE_P. ForwardCore_Tail multiply
 *               opcodes 209 -> 155 (-26%) but total instructions 736 -> 808
 *               (+10%), the balance landing on the ALU pipe. 3090: +0.6% to
 *               +0.9%, faster in 8 of 8 paired reps with non-overlapping
 *               ranges. Small, but it establishes that the multiply pipe is a
 *               real constraint rather than the kernel being purely
 *               issue-bound — while also showing the sensitivity is low, so
 *               cheaper butterflies are not where the remaining headroom is.
 *   2026-08-12  Harvey lazy reduction behind MR_NTT_LAZY. Coefficients ride in
 *               [0,4p) (forward) or [0,2p) (inverse) between butterflies, so
 *               each one costs a single conditional subtract instead of three;
 *               the transform returns to [0,p) only on its last pass.
 *               ForwardCore_Tail SASS 824 -> 736 instructions, of which the
 *               ISETP/SEL pairs went 108 -> 44. Paired A/B on the 3090:
 *               3.384 vs 3.591 ms/bit, i.e. 6.1% FASTER (5.9% faster on the one
 *               rep where both sides happened to run at the same SM clock).
 *   2026-08-11  Added. End to end 11.2% less time, i.e. ~12.6% FASTER, and
 *               12-17% on every one of the seven transform kernels.
 *
 *   Note on units: entries above quote SPEED, so "faster" is always the good
 *   direction. Earlier entries had mixed a time-delta convention with a speed
 *   convention, which made the same sign mean opposite things.
 * ───────────────────────────────────────────────────────────────────────────── */
#pragma once

#include "gpuntt/ntt_merge/ntt.cuh"
#include "config.h"
#include <cstdint>
#include <vector>
#include <stdexcept>
#include <string>

using namespace gpuntt;

// The GPU-NTT merge backend uses a single fixed 64-bit prime for every logn,
// and it happens to be sparse: p = 2^59 + 2^32 - 2^29 + 1 (popcount 5).
#define MR_NTT_MERGE_PRIME 576460756061519873ULL

// q * p taken mod 2^64. With p sparse this is three shifts and three adds
// instead of a 64-bit multiply, moving the work off the multiply pipe (which
// co-limits the transform) onto the ALU pipe (which has headroom). Exact
// identity over the integers, so it needs no range assumptions; the host guard
// in shoup_table_build refuses to run if the prime is ever not this one.
__device__ __forceinline__ Data64 mul_prime_sparse(Data64 q)
{
    return (q << 59) + (q << 32) - (q << 29) + q;
}

__device__ __forceinline__ Data64 mul_prime(Data64 q, Data64 pmod)
{
#ifdef MR_NTT_SPARSE_P
    (void)pmod;
    return mul_prime_sparse(q);
#else
    return q * pmod;
#endif
}

// x * w mod p, with w' = floor(w * 2^64 / p) precomputed. Result in [0, p).
__device__ __forceinline__ Data64 shoup_mulmod(Data64 x, Data64 w, Data64 wq, Data64 pmod)
{
    const Data64 q = __umul64hi(wq, x);
    const Data64 r = w * x - mul_prime(q, pmod);
    return (r >= pmod) ? (r - pmod) : r;
}

#ifdef MR_NTT_LAZY
// Same product without the final conditional subtract: the result is only
// reduced to [0, 2p). Verified for x in [0, 4p): the bound holds and
// r == w*x (mod p) throughout.
__device__ __forceinline__ Data64 shoup_mullazy(Data64 x, Data64 w, Data64 wq, Data64 pmod)
{
    const Data64 q = __umul64hi(wq, x);
    return w * x - mul_prime(q, pmod);
}

// [0, 4p) -> [0, p). Only paid once per coefficient, at the end of the transform.
__device__ __forceinline__ Data64 lazy_final(Data64 x, Data64 pmod)
{
    const Data64 p2 = pmod << 1;
    if (x >= p2)
        x -= p2;
    return (x >= pmod) ? (x - pmod) : x;
}
#endif

// Butterflies taking the twiddle TABLES plus an index, so that the quotient table
// is only dereferenced on the Shoup path (it is null when MR_NTT_SHOUP is off).
__device__ __forceinline__ void CooleyTukeyUnitX(
    Data64 &U, Data64 &V,
    const Root64 *__restrict__ wtab, const Data64 *__restrict__ wqtab,
    unsigned long long idx, const Modulus64 &modulus)
{
#if defined(MR_NTT_LAZY)
    // Harvey: in [0,4p) -> out [0,4p), one conditional subtract instead of three.
    const Data64 pmod = modulus.value;
    const Data64 p2 = pmod << 1;
    Data64 u_ = U;
    if (u_ >= p2)
        u_ -= p2; // [0,2p)
    const Data64 t_ = shoup_mullazy(V, wtab[idx], wqtab[idx], pmod); // [0,2p)
    U = u_ + t_;
    V = u_ - t_ + p2;
#elif defined(MR_NTT_SHOUP)
    const Data64 u_ = U;
    const Data64 v_ = shoup_mulmod(V, wtab[idx], wqtab[idx], modulus.value);
    U = OPERATOR_GPU<Data64>::add(u_, v_, modulus);
    V = OPERATOR_GPU<Data64>::sub(u_, v_, modulus);
#else
    (void)wqtab;
    CooleyTukeyUnit(U, V, wtab[idx], modulus);
#endif
}

__device__ __forceinline__ void GentlemanSandeUnitX(
    Data64 &U, Data64 &V,
    const Root64 *__restrict__ wtab, const Data64 *__restrict__ wqtab,
    unsigned long long idx, const Modulus64 &modulus)
{
#if defined(MR_NTT_LAZY)
    // Harvey: in [0,2p) -> out [0,2p), one conditional subtract instead of three.
    const Data64 pmod = modulus.value;
    const Data64 p2 = pmod << 1;
    Data64 s_ = U + V;
    if (s_ >= p2)
        s_ -= p2; // [0,2p)
    const Data64 d_ = U - V + p2; // [0,4p)
    U = s_;
    V = shoup_mullazy(d_, wtab[idx], wqtab[idx], pmod); // [0,2p)
#elif defined(MR_NTT_SHOUP)
    const Data64 u_ = U;
    const Data64 v_ = V;
    U = OPERATOR_GPU<Data64>::add(u_, v_, modulus);
    const Data64 d = OPERATOR_GPU<Data64>::sub(u_, v_, modulus);
    V = shoup_mulmod(d, wtab[idx], wqtab[idx], modulus.value);
#else
    (void)wqtab;
    GentlemanSandeUnit(U, V, wtab[idx], modulus);
#endif
}

// Scalar variant for the 1/n factor applied by the last inverse pass.
__device__ __forceinline__ Data64 ninv_mul(Data64 x, Ninverse64 n_inverse,
                                           Data64 n_inv_shoup, const Modulus64 &modulus)
{
#ifdef MR_NTT_SHOUP
    return shoup_mulmod(x, (Data64)n_inverse, n_inv_shoup, modulus.value);
#else
    (void)n_inv_shoup;
    return OPERATOR_GPU<Data64>::mult(x, n_inverse, modulus);
#endif
}

// ── host side ────────────────────────────────────────────────────────────────

// w' = floor(w * 2^64 / p) for every twiddle in `roots`.
inline std::vector<Data64> shoup_table_build(const std::vector<Root64> &roots, Data64 p)
{
    if (p >= (Data64)1 << 63)
        throw std::runtime_error(
            "[shoup] NTT prime " + std::to_string(p) +
            " >= 2^63: the r = w*x - q*p step would wrap. Disable MR_NTT_SHOUP.");
#ifdef MR_NTT_SPARSE_P
    // mul_prime_sparse hardcodes this prime's bit pattern; a different prime
    // would silently compute the wrong product.
    if (p != MR_NTT_MERGE_PRIME)
        throw std::runtime_error(
            "[shoup] MR_NTT_SPARSE_P hardcodes p = " +
            std::to_string((Data64)MR_NTT_MERGE_PRIME) + " but the backend chose " +
            std::to_string(p) + ". Disable MR_NTT_SPARSE_P.");
#endif
    std::vector<Data64> out(roots.size());
    for (size_t i = 0; i < roots.size(); i++)
    {
        if ((Data64)roots[i] >= p)
            throw std::runtime_error("[shoup] twiddle >= modulus; table is not reduced.");
        out[i] = (Data64)(((__uint128_t)roots[i] << 64) / (__uint128_t)p);
    }
    return out;
}

inline Data64 shoup_scalar(Data64 w, Data64 p)
{
    return (Data64)(((__uint128_t)w << 64) / (__uint128_t)p);
}
