#pragma once
// Runs a benchmark of Montgomery operations GPU vs GMP and prints a table.
// Enabled with --bench-ops (up to 65536 bits) or --bench-ops-long (up to 131072 bits).
void run_bench_ops(bool long_run = false);
