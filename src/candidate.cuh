#pragma once
// candidate.cuh — Represents prime candidates for the Miller-Rabin test.
//
// NumberCandidate: holds N, N-1, and the odd part d of N-1 as limb arrays.
// GroupInfo:       label + equation strings for a group; no built state.
// LazyCandidate:   one equation that evaluates and builds its limb arrays on
//                  demand (first call to get()) and caches the result.

#include <vector>
#include <string>
#include <memory>
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

struct NumberCandidate {
    int s = 0;
    std::vector<uint64_t> N_lims;
    std::vector<uint64_t> Nm1_lims;
    std::vector<uint64_t> d_lims;

    bool is_s1() const { return s == 1; }

    void build_from_mpz(const mpz_t N, int n_limbs)
    {
        N_lims.assign(n_limbs, 0);
        Nm1_lims.assign(n_limbs, 0);
        d_lims.assign(n_limbs, 0);

        mpz_t Nm1, d;
        mpz_inits(Nm1, d, nullptr);
        mpz_sub_ui(Nm1, N, 1);

        s = 0;
        mpz_set(d, Nm1);
        while (mpz_even_p(d)) { mpz_tdiv_q_2exp(d, d, 1); s++; }

        mpz_to_limbs_vec(N_lims.data(),   n_limbs, N);
        mpz_to_limbs_vec(Nm1_lims.data(), n_limbs, Nm1);
        mpz_to_limbs_vec(d_lims.data(),   n_limbs, d);

        mpz_clears(Nm1, d, nullptr);
    }
};

// ── Group metadata ────────────────────────────────────────────────────────────
// Just the label and equation strings; no build state.

struct GroupInfo {
    std::string label;
    std::vector<std::string> equations;
};

// ── Lazy candidate ────────────────────────────────────────────────────────────
// Holds one equation and its build cache.  Non-copyable, movable.
// Evaluating the equation (mpz) happens on the first call to natural_n_limbs()
// or get().  The NumberCandidate limb arrays are built on the first get(n_limbs)
// call and rebuilt only if n_limbs changes (rare: only when a new batch has a
// larger candidate).

struct LazyCandidate {
    std::string equation;
    int group_idx = -1;
    int round_idx = -1;

    LazyCandidate() = default;
    LazyCandidate(const LazyCandidate &) = delete;
    LazyCandidate &operator=(const LazyCandidate &) = delete;
    LazyCandidate(LazyCandidate &&o) noexcept
        : equation(std::move(o.equation)),
          group_idx(o.group_idx), round_idx(o.round_idx),
          mpz_ready_(o.mpz_ready_), built_n_limbs_(o.built_n_limbs_),
          built_(std::move(o.built_))
    {
        if (mpz_ready_) {
            mpz_init_set(val_, o.val_);
            mpz_clear(o.val_);
            o.mpz_ready_ = false;
        }
    }
    LazyCandidate &operator=(LazyCandidate &&o) noexcept
    {
        if (this != &o) {
            free_all();
            equation       = std::move(o.equation);
            group_idx      = o.group_idx;
            round_idx      = o.round_idx;
            built_n_limbs_ = o.built_n_limbs_;
            built_         = std::move(o.built_);
            mpz_ready_     = o.mpz_ready_;
            if (mpz_ready_) {
                mpz_init_set(val_, o.val_);
                mpz_clear(o.val_);
                o.mpz_ready_ = false;
            }
        }
        return *this;
    }
    ~LazyCandidate() { free_all(); }

    // Returns the minimum n_limbs dictated by this candidate's digit count.
    int natural_n_limbs() const
    {
        ensure_val();
        return limbs_for_digits((int)mpz_sizeinbase(val_, 10) + 4);
    }

    // Returns the built NumberCandidate for the given n_limbs.
    // Rebuilds if n_limbs differs from the cached value (zero-extension is safe).
    const NumberCandidate &get(int n_limbs) const
    {
        ensure_val();
        if (built_n_limbs_ != n_limbs) {
            built_.build_from_mpz(val_, n_limbs);
            built_n_limbs_ = n_limbs;
        }
        return built_;
    }

    // Release everything (mpz + limb arrays).
    void free_all()
    {
        if (mpz_ready_) { mpz_clear(val_); mpz_ready_ = false; }
        built_.N_lims.clear();   built_.N_lims.shrink_to_fit();
        built_.Nm1_lims.clear(); built_.Nm1_lims.shrink_to_fit();
        built_.d_lims.clear();   built_.d_lims.shrink_to_fit();
        built_n_limbs_ = 0;
    }

private:
    mutable mpz_t         val_;
    mutable bool          mpz_ready_     = false;
    mutable int           built_n_limbs_ = 0;
    mutable NumberCandidate built_;

    void ensure_val() const
    {
        if (mpz_ready_) return;
        mpz_init(val_);
        EquationParser::eval(equation, val_);
        if (mpz_sgn(val_) <= 0)
            throw std::runtime_error(
                "equation \"" + equation + "\" evaluated to a non-positive value");
        mpz_ready_ = true;
    }
};

// ── pack_batch ────────────────────────────────────────────────────────────────

inline void pack_batch(
    const std::vector<const NumberCandidate *> &cands,
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

// Overload for mutable pointers (backward compat with correctness tests).
inline void pack_batch(
    const std::vector<NumberCandidate *> &cands,
    int n_limbs,
    std::vector<uint64_t> &N_out,
    std::vector<uint64_t> &Nm1_out,
    std::vector<uint64_t> &d_out)
{
    std::vector<const NumberCandidate *> cv(cands.begin(), cands.end());
    pack_batch(cv, n_limbs, N_out, Nm1_out, d_out);
}
