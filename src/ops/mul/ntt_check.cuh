#pragma once
// ops/mul/ntt_check.cuh — spec checks shared by the NTT backends.
// Include AFTER the backend header (needs LIMB_MASK).

#include <stdexcept>
#include <string>
#include <cstdint>

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
    const __uint128_t lm = (__uint128_t)LIMB_MASK; // 2^LIMB_BITS − 1
    const __uint128_t max_coeff = max_terms * lm * lm;
    if (max_coeff >= (__uint128_t)p_val)
        throw std::runtime_error(
            "[ntt] insufficient precision: (padded/2)*(2^LIMB_BITS-1)^2 (~"
            "terms=" + std::to_string(padded / 2) + ", LIMB_BITS=" + std::to_string(LIMB_BITS) +
            ") >= NTT prime (" + std::to_string(p_val) +
            "). Reduce LIMB_BITS or the operand size.");
}
