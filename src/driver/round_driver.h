/* ─────────────────────────────────────────────────────────────────────────────
 * FILE   src/driver/round_driver.h
 * ROLE   round-based orchestration of a candidate file
 *
 * HOW    See round_driver.cu.
 *
 * CHANGELOG
 *   2026-08-11  Extracted from bench_mr_gpu.cu.
 * ───────────────────────────────────────────────────────────────────────────── */
#pragma once

#include "driver/input_parser.h"
#include <vector>

/* Runs every group in `groups` to a verdict and prints the report.
 *
 * PARAMS
 *   groups         parsed input, one entry per group
 *   show_report    print the per-candidate detail table
 *   show_progress  print the live progress bar
 */
void run_rounds(std::vector<GroupInfo> &groups, bool show_report, bool show_progress);
