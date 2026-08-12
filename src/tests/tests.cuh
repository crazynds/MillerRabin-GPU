/* ─────────────────────────────────────────────────────────────────────────────
 * FILE   src/tests/tests.cuh
 * ROLE   entry points of the GMP-checked correctness suites
 *
 * HOW    Each suite lives in its own translation unit under tests/ and is
 *        reachable only through these four declarations. They run when the
 *        binary is given --test.
 *
 * CHANGELOG
 *   2026-08-11  Extracted from correctness_tests.cuh (1082 lines, all in a
 *               header).
 * ───────────────────────────────────────────────────────────────────────────── */
#pragma once

/* Compares every GPU modular primitive against GMP on random operands. */
void run_correctness_tests();

/* Checks that known Mersenne primes are reported prime. */
void run_known_prime_tests();

/* Checks primes whose N-1 has s != 1, exercising the general MR path. */
void run_general_s_prime_tests();

/* Checks s == 1 primes produced by mpz_nextprime. */
void run_s1_nextprime_tests();
