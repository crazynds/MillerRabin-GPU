/* ─────────────────────────────────────────────────────────────────────────────
 * FILE   src/bench/bench_ops.cuh
 * ROLE   entry point of --bench-ops
 *
 * HOW    Sweeps operand sizes and prints a GPU-vs-GMP table of the modular
 *        primitives. long_run extends the sweep from 65536 to 131072 bits.
 * ───────────────────────────────────────────────────────────────────────────── */
#pragma once

void run_bench_ops(bool long_run = false);
