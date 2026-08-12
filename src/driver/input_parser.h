/* ─────────────────────────────────────────────────────────────────────────────
 * FILE   src/driver/input_parser.h
 * ROLE   reads the candidate file into groups
 *
 * HOW    See input_parser.cu.
 * ───────────────────────────────────────────────────────────────────────────── */
#pragma once

#include "driver/candidate.cuh"
#include <vector>

std::vector<GroupInfo> parse_input(const char *path);
