#pragma once
#include "candidate.cuh"
#include <vector>
#include <string>

std::pair<bool, double> cpu_test_equation(const std::string &equation);
void run_cpu_mode(std::vector<GroupInfo> &groups, bool show_report, bool show_progress, int n_threads);
