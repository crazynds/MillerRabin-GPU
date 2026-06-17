#pragma once
// helpers/gmp_helpers.cuh — host conversions between limbs (base 2^LIMB_BITS, little-endian) and mpz_t.
// Shared by batch_mod_ctx.cu and the reduction files (reductions/*.cu).
// Requires LIMB_BITS/LIMB_MASK (config.h + backend) already included before this header.

#include <cstdint>
#include <gmp.h>

// limbs little-endian (base 2^LIMB_BITS) → mpz_t.
static inline void limbs_to_mpz(mpz_t out, const uint64_t *lims, int n)
{
    mpz_set_ui(out, 0);
    for (int i = n - 1; i >= 0; i--)
    {
        mpz_mul_2exp(out, out, LIMB_BITS);
        mpz_add_ui(out, out, (unsigned long)lims[i]);
    }
}

// mpz_t → n little-endian limbs (base 2^LIMB_BITS) (truncates above n).
static inline void mpz_to_limbs(uint64_t *out, int n, const mpz_t x)
{
    mpz_t tmp;
    mpz_init_set(tmp, x);
    for (int i = 0; i < n; i++)
    {
        out[i] = mpz_get_ui(tmp) & LIMB_MASK;
        mpz_tdiv_q_2exp(tmp, tmp, LIMB_BITS);
    }
    mpz_clear(tmp);
}
