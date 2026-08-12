/* ─────────────────────────────────────────────────────────────────────────────
 * FILE   src/util/gmp_helpers.cuh
 * ROLE   host conversions between limb arrays and mpz_t
 *
 * HOW    Both directions go through a flat little-endian byte buffer
 *        (mpz_export / mpz_import), which makes them O(n).
 *
 * NOTE   The obvious loop — shifting the mpz_t by LIMB_BITS once per limb —
 *        is O(n^2): every iteration touches the whole number. At 16k limbs
 *        that is 19 ms per conversion against 0.15 ms here, and the setup
 *        path runs hundreds of them.
 *
 * CHANGELOG
 *   2026-08-11  Rewritten in O(n). The three duplicate copies of the
 *               quadratic loop (here, candidate.cuh, bench_ops.cu) collapsed
 *               into this one.
 * ───────────────────────────────────────────────────────────────────────────── */
#pragma once

#include "ops/mul/multiplier.cuh" /* LIMB_BITS / LIMB_MASK */
#include <cstdint>
#include <vector>
#include <gmp.h>

static_assert(LIMB_BITS >= 1 && LIMB_BITS <= 32,
              "gmp_helpers: LIMB_BITS must fit the 8-byte gather window used below");

// limbs little-endian (base 2^LIMB_BITS) → mpz_t.
static inline void limbs_to_mpz(mpz_t out, const uint64_t *lims, int n)
{
    if (n <= 0)
    {
        mpz_set_ui(out, 0);
        return;
    }
    const size_t nbytes = ((size_t)n * LIMB_BITS + 7) / 8 + 8;
    std::vector<unsigned char> buf(nbytes, 0);

    for (int i = 0; i < n; i++)
    {
        uint64_t cur = (lims[i] & LIMB_MASK);
        if (!cur)
            continue;
        const size_t bit = (size_t)i * LIMB_BITS;
        size_t by = bit >> 3;
        cur <<= (bit & 7);
        while (cur)
        {
            buf[by++] |= (unsigned char)(cur & 0xFFu);
            cur >>= 8;
        }
    }
    mpz_import(out, nbytes, -1 /*least significant word first*/, 1 /*byte words*/,
               -1 /*little endian*/, 0 /*no nails*/, buf.data());
}

// mpz_t → n little-endian limbs (base 2^LIMB_BITS) (truncates above n).
static inline void mpz_to_limbs(uint64_t *out, int n, const mpz_t x)
{
    if (n <= 0)
        return;
    const size_t xbytes = (mpz_sizeinbase(x, 2) + 7) / 8;
    const size_t need = ((size_t)n * LIMB_BITS + 7) / 8;
    // +8 slack: the gather below always reads a full 8-byte window.
    std::vector<unsigned char> buf((xbytes > need ? xbytes : need) + 8, 0);

    if (mpz_sgn(x) != 0)
    {
        size_t written = 0;
        mpz_export(buf.data(), &written, -1, 1, -1, 0, x);
    }

    for (int i = 0; i < n; i++)
    {
        const size_t bit = (size_t)i * LIMB_BITS;
        const unsigned char *p = buf.data() + (bit >> 3);
        uint64_t v = 0;
        for (int k = 0; k < 8; k++)
            v |= (uint64_t)p[k] << (8 * k);
        out[i] = (v >> (bit & 7)) & LIMB_MASK;
    }
}

// Compares two limb arrays of the same length (little-endian, base 2^LIMB_BITS).
// Returns <0, 0 or >0 like memcmp. O(n), no GMP involved.
static inline int limbs_cmp(const uint64_t *a, const uint64_t *b, int n)
{
    for (int i = n - 1; i >= 0; i--)
    {
        if (a[i] != b[i])
            return (a[i] < b[i]) ? -1 : 1;
    }
    return 0;
}
