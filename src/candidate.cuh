#pragma once
// candidate.cuh — Represents prime candidates for the Miller-Rabin test.
//
// NumberCandidate: holds N, N-1, and the odd part d of N-1 as limb arrays.
//                  Constructed from any mpz_t via build_from_mpz().
// GroupCandidate:  a list of NumberCandidates belonging to the same test group.
//                  Each entry is built from one equation string. Within a group,
//                  candidates are tested in order; the group fails as soon as any
//                  candidate is composite (the remaining ones are skipped).

#include <vector>
#include <string>
#include <stdexcept>
#include <gmp.h>
#include "ops/mul/multiplier.cuh"
#include "equation.h"

// ── GMP utilities ────────────────────────────────────────────────────────────

static inline void mpz_to_limbs_vec(uint64_t *out, int n, const mpz_t x)
{
    mpz_t tmp;
    mpz_init_set(tmp, x);
    for (int i = 0; i < n; i++) {
        out[i] = mpz_get_ui(tmp) & LIMB_MASK;
        mpz_tdiv_q_2exp(tmp, tmp, LIMB_BITS);
    }
    mpz_clear(tmp);
}

// ── Single candidate ─────────────────────────────────────────────────────────
//
// Represents N with its decomposition N - 1 = 2^s * d  (d odd).

struct NumberCandidate {
    int s = 0; // N - 1 = 2^s * d

    std::vector<uint64_t> N_lims;
    std::vector<uint64_t> Nm1_lims;
    std::vector<uint64_t> d_lims; // d such that N-1 = 2^s * d, d odd

    bool is_s1() const { return s == 1; }

    void build_from_mpz(const mpz_t N, int n_limbs)
    {
        N_lims.assign(n_limbs, 0);
        Nm1_lims.assign(n_limbs, 0);
        d_lims.assign(n_limbs, 0);

        mpz_t Nm1, d;
        mpz_inits(Nm1, d, nullptr);
        mpz_sub_ui(Nm1, N, 1);

        // Find s: largest power of 2 dividing N-1
        s = 0;
        mpz_set(d, Nm1);
        while (mpz_even_p(d)) {
            mpz_tdiv_q_2exp(d, d, 1);
            s++;
        }

        mpz_to_limbs_vec(N_lims.data(),   n_limbs, N);
        mpz_to_limbs_vec(Nm1_lims.data(), n_limbs, Nm1);
        mpz_to_limbs_vec(d_lims.data(),   n_limbs, d);

        mpz_clears(Nm1, d, nullptr);
    }
};

// ── Group of candidates ───────────────────────────────────────────────────────
//
// A group groups related equations (e.g. N and its digit-reversed twin revN)
// under a common label. Candidates are tested round-by-round: if round k is
// composite the group is eliminated and round k+1 is never dispatched to the GPU.
//
// All candidates in a group share n_limbs so they can be co-batched.

struct GroupCandidate {
    std::string label;                  // user-supplied group ID (may be empty)
    std::vector<std::string> equations; // raw equation strings (for display)
    std::vector<NumberCandidate> cands; // one per equation, after build()
    int n_limbs = 0;

    // Evaluate every equation with GMP and construct the limb arrays.
    // All numbers are sized to the widest equation in the group.
    void build()
    {
        size_t n = equations.size();
        std::vector<__mpz_struct> ns(n);
        int max_digits = 0;

        for (size_t i = 0; i < n; i++) {
            mpz_init(&ns[i]);
            EquationParser::eval(equations[i], &ns[i]);
            if (mpz_sgn(&ns[i]) <= 0)
                throw std::runtime_error(
                    "equation \"" + equations[i] + "\" evaluated to a non-positive value");
            int d = (int)mpz_sizeinbase(&ns[i], 10);
            if (d > max_digits) max_digits = d;
        }

        n_limbs = limbs_for_digits(max_digits + 4);
        cands.resize(n);
        for (size_t i = 0; i < n; i++) {
            cands[i].build_from_mpz(&ns[i], n_limbs);
            mpz_clear(&ns[i]);
        }
    }
};

inline void pack_batch(
    const std::vector<NumberCandidate *> &cands,
    int n_limbs,
    std::vector<uint64_t> &N_out,
    std::vector<uint64_t> &Nm1_out,
    std::vector<uint64_t> &d_out)
{
    int bsz = (int)cands.size();
    N_out.assign((size_t)bsz * n_limbs, 0);
    Nm1_out.assign((size_t)bsz * n_limbs, 0);
    d_out.assign((size_t)bsz * n_limbs, 0);
    for (int i = 0; i < bsz; i++) {
        std::copy(cands[i]->N_lims.begin(),   cands[i]->N_lims.end(),   N_out.begin()   + i * n_limbs);
        std::copy(cands[i]->Nm1_lims.begin(), cands[i]->Nm1_lims.end(), Nm1_out.begin() + i * n_limbs);
        std::copy(cands[i]->d_lims.begin(),   cands[i]->d_lims.end(),   d_out.begin()   + i * n_limbs);
    }
}
