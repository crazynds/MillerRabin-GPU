/* ─────────────────────────────────────────────────────────────────────────────
 * FILE   src/driver/cpu_runner.h
 * ROLE   entry point of --cpu / --threads
 *
 * HOW    See cpu_runner.cu.
 * ───────────────────────────────────────────────────────────────────────────── */
#pragma once

#include "driver/candidate.cuh"
#include <vector>
#include <string>

std::pair<bool, double> cpu_test_equation(const std::string &equation);
void run_cpu_mode(std::vector<GroupInfo> &groups, bool show_report, bool show_progress, int n_threads);
