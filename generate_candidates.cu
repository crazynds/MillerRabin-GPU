// generate_candidates.cu — GPU prime candidate sieve
//
// Candidates:  N  = 10^n - K*10^B - J*10^C - 1
//              N' = digit_reverse(N)
//              = 10^n - rev(K)*10^(n-B-d(K)) - rev(J)*10^(n-C-d(J)) - 1
//
// Approach:
//   Keep an "alive[65536]" array (one slot per (K,J) pair, K,J in 1..256).
//   For each batch of primes p:
//     - Thread 0 in each warp precomputes N mod p residues into shared memory.
//     - Every candidate thread checks its (K,J): if K*a + J*b ≡ T mod p (or N'≡0),
//       mark dead.  No modular inverse needed — just multiply and compare.
//
// Build:
//   nvcc -O3 --arch=native generate_candidates.cu -o gen_candidates
//
// Usage:
//   ./gen_candidates [-n 50000] [-t 256] [-p 32] [-o output.txt] [B C [B C ...]]

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <chrono>

// ── Constants ─────────────────────────────────────────────────────────────────
static constexpr int     NKMAX   = 256;
static constexpr int     NCANDS  = NKMAX * NKMAX;   // 65 536
static constexpr uint32_t SEG_SZ = 1u << 22;        // 4 M numbers per CPU seg
static constexpr int     BATCH   = 1 << 18;         // 256 K primes per GPU batch
static constexpr int     BLOCK   = 256;             // threads per block

// ── CUDA helpers ──────────────────────────────────────────────────────────────

__device__ __forceinline__
uint64_t powmod64(uint64_t base, uint64_t exp, uint64_t mod)
{
    uint64_t r = 1; base %= mod;
    while (exp) { if (exp & 1) r = r * base % mod; base = base * base % mod; exp >>= 1; }
    return r;
}
__device__ __forceinline__ int dev_rev(int x) { int r=0; while(x){r=r*10+x%10;x/=10;} return r; }
__device__ __forceinline__ int dev_digs(int x) { return x<10?1:x<100?2:3; }

// ── GPU Kernel ────────────────────────────────────────────────────────────────
//
// Grid : NCANDS/BLOCK blocks × BLOCK threads
// Each block processes all primes in the batch.
// Thread 0 in each block precomputes residues for each prime into shared memory;
// all threads read them and check their own (K,J) candidate.

__global__ void sieve_kernel(
    const uint32_t* __restrict__ primes, int np,
    uint32_t B, uint32_t C, uint32_t n,
    uint8_t* __restrict__ alive          // [NCANDS]
)
{
    // Shared: mod values for the current prime, computed once per block.
    __shared__ uint64_t sh_p;
    __shared__ uint64_t sh_T;          // (10^n - 1) mod p
    __shared__ uint64_t sh_a;          // 10^B mod p
    __shared__ uint64_t sh_b;          // 10^C mod p
    __shared__ uint64_t sh_ar[3];      // 10^(n-B-d) mod p, d = 1,2,3
    __shared__ uint64_t sh_br[3];      // 10^(n-C-d) mod p, d = 1,2,3

    int cid = blockIdx.x * blockDim.x + threadIdx.x;

    // Candidate-specific values (computed once, reused across all primes)
    int K=0, J=0, Kp=0, Jp=0, dk=0, dj=0;
    bool alive_flag = false;
    if (cid < NCANDS) {
        K  = cid / NKMAX + 1;  J  = cid % NKMAX + 1;
        Kp = dev_rev(K);        Jp = dev_rev(J);
        dk = dev_digs(K);       dj = dev_digs(J);
        alive_flag = (bool)alive[cid];
    }

    for (int pi = 0; pi < np; pi++) {

        // ── Thread 0: precompute shared residues for prime primes[pi] ──────
        if (threadIdx.x == 0) {
            uint64_t p  = primes[pi];
            sh_p        = p;
            sh_T        = (powmod64(10, n, p) + p - 1) % p;
            sh_a        = powmod64(10, B, p);
            sh_b        = powmod64(10, C, p);
            // 10^(n-B-d) and 10^(n-C-d) for d = 1,2,3
            for (int d = 1; d <= 3; d++) {
                int eA = (int)n - (int)B - d;
                int eC = (int)n - (int)C - d;
                sh_ar[d-1] = (eA >= 0) ? powmod64(10, (uint64_t)eA, p) : 0;
                sh_br[d-1] = (eC >= 0) ? powmod64(10, (uint64_t)eC, p) : 0;
            }
        }
        __syncthreads();   // all threads see sh_* before proceeding

        // ── Every candidate checks itself ─────────────────────────────────
        if (cid < NCANDS && alive_flag) {
            uint64_t p = sh_p;

            // N = 10^n - K*10^B - J*10^C - 1  ≡ 0 (mod p)?
            // ⟺  K*a + J*b  ≡  T  (mod p)
            uint64_t valN = ((uint64_t)K * sh_a + (uint64_t)J * sh_b) % p;
            if (valN == sh_T) {
                alive_flag = false;
            } else {
                // N' = 10^n - rev(K)*10^(n-B-dk) - rev(J)*10^(n-C-dj) - 1
                uint64_t ar = sh_ar[dk - 1];
                uint64_t br = sh_br[dj - 1];
                if (ar && br) {
                    uint64_t valNp = ((uint64_t)Kp * ar + (uint64_t)Jp * br) % p;
                    if (valNp == sh_T) alive_flag = false;
                }
            }
        }
        __syncthreads();   // thread 0 must not overwrite sh_* before others finish
    }

    if (cid < NCANDS && !alive_flag) alive[cid] = 0;
}

// ── CPU: segmented sieve of Eratosthenes ─────────────────────────────────────

static std::vector<uint32_t> g_small;   // primes 3..65521, excluding 2 and 5

static void init_small_primes()
{
    const int lim = 65537;
    std::vector<bool> sv(lim, true);
    sv[0] = sv[1] = false;
    for (int i = 2; (long long)i*i < lim; i++)
        if (sv[i]) for (int j = i*i; j < lim; j += i) sv[j] = false;
    for (int i = 3; i < lim; i++)
        if (sv[i] && i != 5) g_small.push_back((uint32_t)i);
}

static void seg_sieve(uint64_t lo, uint64_t hi, std::vector<uint32_t>& out)
{
    size_t sz = (size_t)(hi - lo);
    std::vector<uint8_t> sv(sz, 1);
    for (uint32_t sp : g_small) {
        uint64_t start = ((lo + sp - 1) / sp) * (uint64_t)sp;
        if (start < (uint64_t)sp * sp) start = (uint64_t)sp * sp;
        for (uint64_t j = start; j < hi; j += sp) sv[(size_t)(j-lo)] = 0;
    }
    for (size_t i = 0; i < sz; i++) {
        uint64_t v = lo + i;
        if (sv[i] && v > 5 && v <= 0xFFFFFFFFull) out.push_back((uint32_t)v);
    }
}

// ── CUDA error check ──────────────────────────────────────────────────────────
#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1); } } while(0)

// ── Run sieve for one (B, C) pair ─────────────────────────────────────────────

static void run_sieve(uint32_t B, uint32_t C, uint32_t n,
                      uint64_t p_limit, uint8_t* alive_host)
{
    uint8_t* d_alive;
    uint32_t* d_primes;
    CK(cudaMalloc(&d_alive,  NCANDS * sizeof(uint8_t)));
    CK(cudaMalloc(&d_primes, BATCH  * sizeof(uint32_t)));
    CK(cudaMemcpy(d_alive, alive_host, NCANDS, cudaMemcpyHostToDevice));

    int blocks = (NCANDS + BLOCK - 1) / BLOCK;

    // Lambda: upload primes batch and launch kernel
    auto flush = [&](const uint32_t* buf, int np) {
        CK(cudaMemcpy(d_primes, buf, (size_t)np * sizeof(uint32_t), cudaMemcpyHostToDevice));
        sieve_kernel<<<blocks, BLOCK>>>(d_primes, np, B, C, n, d_alive);
    };

    std::vector<uint32_t> buf;
    buf.reserve(BATCH + SEG_SZ);

    auto maybe_flush = [&]() {
        while ((int)buf.size() >= BATCH) {
            flush(buf.data(), BATCH);
            buf.erase(buf.begin(), buf.begin() + BATCH);
        }
    };

    // Phase 1: small primes (already in g_small)
    buf.insert(buf.end(), g_small.begin(), g_small.end());
    maybe_flush();

    // Phase 2: large primes via segmented sieve
    auto t0 = std::chrono::steady_clock::now();
    uint64_t lo = (uint64_t)g_small.back() + 1;
    int seg_idx = 0;
    for (; lo < p_limit; lo += SEG_SZ) {
        seg_sieve(lo, std::min(lo + (uint64_t)SEG_SZ, p_limit), buf);
        maybe_flush();
        if (++seg_idx % 128 == 0) {
            double frac = (double)(lo - g_small.back()) / (p_limit - g_small.back());
            double dt   = std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
            fprintf(stderr, "\r  %.1f%%  (%.0fs)   ", 100.0*frac, dt);
            fflush(stderr);
        }
    }
    if (!buf.empty()) flush(buf.data(), (int)buf.size());

    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(alive_host, d_alive, NCANDS, cudaMemcpyDeviceToHost));

    double dt = std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
    int survivors = 0;
    for (int i = 0; i < NCANDS; i++) survivors += alive_host[i];
    fprintf(stderr, "\r  B=%u C=%u: %d survivors  (%.1fs)\n", B, C, survivors, dt);

    CK(cudaFree(d_alive));
    CK(cudaFree(d_primes));
}

// ── CPU helpers ───────────────────────────────────────────────────────────────
static int cpu_rev(int x)  { int r=0; while(x){r=r*10+x%10;x/=10;} return r; }
static int cpu_digs(int x) { return x<10?1:x<100?2:3; }

// ── Main ──────────────────────────────────────────────────────────────────────

int main(int argc, char** argv)
{
    int      n_exp   = 50000;
    int      target  = 256;
    int      p_bits  = 32;
    const char* out_file = nullptr;
    std::vector<std::pair<int,int>> bc_pairs;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i],"-n") && i+1<argc)  n_exp   = atoi(argv[++i]);
        else if (!strcmp(argv[i],"-t") && i+1<argc) target  = atoi(argv[++i]);
        else if (!strcmp(argv[i],"-p") && i+1<argc) p_bits  = atoi(argv[++i]);
        else if (!strcmp(argv[i],"-o") && i+1<argc) out_file = argv[++i];
        else if (i+1 < argc) {
            int B = atoi(argv[i]), C = atoi(argv[i+1]);
            if (B > C && B > 0 && C > 0) { bc_pairs.push_back({B, C}); i++; }
        }
    }

    const uint32_t n       = (uint32_t)n_exp;
    const uint64_t p_limit = (uint64_t)1 << p_bits;

    // Print GPU info
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        fprintf(stderr, "GPU: %s  (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    }
    fprintf(stderr, "n=%u  target=%d  sieve up to 2^%d\n", n, target, p_bits);

    init_small_primes();
    fprintf(stderr, "Small primes: %zu  (3..%u)\n\n", g_small.size(), g_small.back());

    // Auto-select (B, C) if none given
    if (bc_pairs.empty()) {
        double fb[] = {0.20,0.30,0.40,0.50,0.60,0.70,0.80};
        double fc[] = {0.04,0.08,0.12,0.16};
        for (double b : fb) for (double c : fc) {
            int B = (int)(n*b), C = (int)(n*c);
            if (B > C+1 && B+C != (int)n-1 && C >= 4 && B <= (int)n-4)
                bc_pairs.push_back({B,C});
        }
        std::mt19937 rng(42);
        while ((int)bc_pairs.size() < 60) {
            int B = 4 + rng() % (n - 8);
            int C = 4 + rng() % std::max(1u, (unsigned)B - 4);
            if (B > C && B+C != (int)n-1 && B <= (int)n-4) bc_pairs.push_back({B,C});
        }
    }

    FILE* out = out_file ? fopen(out_file,"w") : stdout;
    if (!out) { perror("fopen"); return 1; }

    std::vector<uint8_t> alive(NCANDS);
    int n_collected = 0;

    for (auto [B, C] : bc_pairs) {
        if (n_collected >= target) break;

        fprintf(stderr, "[%d/%d]  B=%d C=%d\n", n_collected, target, B, C);

        // Reset all candidates to alive
        std::fill(alive.begin(), alive.end(), 1);

        run_sieve((uint32_t)B, (uint32_t)C, n, p_limit, alive.data());

        // Collect surviving (K, J) pairs
        for (int cid = 0; cid < NCANDS && n_collected < target; cid++) {
            if (!alive[cid]) continue;

            int K = cid / NKMAX + 1, J = cid % NKMAX + 1;
            int lk = cpu_digs(K), lj = cpu_digs(J);
            int Kp = cpu_rev(K),  Jp = cpu_rev(J);
            int Bp = n - B - lk,  Cp = n - C - lj;
            if (Bp < 0 || Cp < 0) continue;

            char lbl[64];
            snprintf(lbl, sizeof(lbl), "g%d_%d_%d_%d", B, C, K, J);

            // N  = 10^n - K*10^B - J*10^C - 1   (B > C by construction)
            fprintf(out, "%s: 10^%u - %d*10^%d - %d*10^%d - 1\n", lbl, n, K, B, J, C);

            // N' — write with larger exponent first
            if (Bp >= Cp)
                fprintf(out, "%s: 10^%u - %d*10^%d - %d*10^%d - 1\n", lbl, n, Kp, Bp, Jp, Cp);
            else
                fprintf(out, "%s: 10^%u - %d*10^%d - %d*10^%d - 1\n", lbl, n, Jp, Cp, Kp, Bp);

            n_collected++;
        }
    }

    fprintf(stderr, "\nWritten %d groups (%d equations).\n", n_collected, n_collected * 2);
    if (out != stdout) fclose(out);
    return 0;
}
