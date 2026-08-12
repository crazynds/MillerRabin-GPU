/* ─────────────────────────────────────────────────────────────────────────────
 * FILE   src/ops/mul/ntt_check.cuh
 * ROLE   runtime guards shared by the NTT backends
 *
 * HOW    check_ntt_precision refuses a configuration whose largest possible
 *        coefficient, (padded/2)*(2^LIMB_BITS-1)^2, could reach the NTT
 *        prime — silent wraparound would otherwise give a wrong product
 *        with no symptom. check_ntt_padding_efficiency warns when the
 *        transform is mostly padding and names the LIMB_BITS that would
 *        avoid it.
 *
 * CHANGELOG
 *   2026-08-11  Added check_ntt_padding_efficiency. Landing 2*n_limbs just
 *               above a power of two doubles every transform for nothing; at
 *               100k digits that was a 1.89x loss going unnoticed.
 * ───────────────────────────────────────────────────────────────────────────── */
#pragma once

#include <stdexcept>
#include <string>
#include <cstdint>
#include <cstdio>

// Precision guarantee of the integer NTT over base 2^LIMB_BITS.
//
// The convolution produces coefficients = Σ A[i]·B[k-i]. The number of summed terms
// is, in the worst case of this system, ≤ padded/2 (operands occupy ≤ n_limbs ≤
// padded/2 limbs, since padded ≥ 2·n_limbs). Each term ≤ (2^LIMB_BITS − 1)². If the
// largest possible coefficient reaches the NTT prime, there is silent wraparound →
// wrong product. We throw a clear error instead of producing garbage.
inline void check_ntt_precision(int padded, unsigned long long p_val)
{
    const __uint128_t max_terms = (__uint128_t)(padded / 2);
    const __uint128_t lm = (__uint128_t)LIMB_MASK;
    const __uint128_t max_coeff = max_terms * lm * lm;
    if (max_coeff >= (__uint128_t)p_val)
        throw std::runtime_error(
            "[ntt] insufficient precision: (padded/2)*(2^LIMB_BITS-1)^2 (~"
            "terms=" + std::to_string(padded / 2) + ", LIMB_BITS=" + std::to_string(LIMB_BITS) +
            ") >= NTT prime (" + std::to_string(p_val) +
            "). Reduce LIMB_BITS or the operand size.");
}

// Padding efficiency guard.
//
// `padded` is the next power of two at or above 2*n_limbs, so a LIMB_BITS that
// makes 2*n_limbs land just *above* a power of two costs a full 2x in every
// transform — and transforms are ~80% of GPU time. The operand size in bits is
// fixed by the candidate, so the only free variable is LIMB_BITS: a larger limb
// means fewer limbs, and may drop the transform a whole octave.
//
// Example measured on a 3090 at 100k decimal digits: LIMB_BITS=20 gives
// 2*n_limbs = 33234, just 1.4% over 2^15, forcing padded = 2^16 (50.7% used).
// LIMB_BITS=22 gives 30214 -> padded = 2^15 and the whole test runs 1.89x faster.
//
// This only warns; LIMB_BITS is a compile-time constant, so acting on it means a
// rebuild. Warned once per process.
inline void check_ntt_padding_efficiency(int n_limbs, int padded, unsigned long long p_val)
{
    static bool warned = false;
    if (warned || padded <= 0)
        return;
    if ((double)(2 * n_limbs) / (double)padded >= 0.6)
        return;

    const long long bits = (long long)n_limbs * LIMB_BITS;

    int best_b = 0, best_padded = padded;
    for (int b = LIMB_BITS + 1; b <= 32; b++)
    {
        const long long nl = (bits + b - 1) / b + 5;
        long long p2 = 1;
        while (p2 < 2 * nl)
            p2 <<= 1;
        const __uint128_t lm = (__uint128_t)((1ULL << b) - 1ULL);
        if ((__uint128_t)(p2 / 2) * lm * lm >= (__uint128_t)p_val)
            continue;
        if (p2 < best_padded || (best_b && p2 == best_padded))
        {
            best_padded = (int)p2;
            best_b = b;
        }
    }
    if (!best_b)
        return;

    warned = true;
    fprintf(stderr,
            "[ntt] warning: transform is %.0f%% padding — 2*n_limbs=%d but padded=%d.\n"
            "      LIMB_BITS=%d would reach padded=%d (%.1fx less transform work).\n"
            "      Set it in params.cmake and rebuild.\n",
            100.0 * (1.0 - (double)(2 * n_limbs) / (double)padded),
            2 * n_limbs, padded, best_b, best_padded,
            (double)padded / (double)best_padded);
}
