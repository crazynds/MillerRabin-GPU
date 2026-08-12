/* ─────────────────────────────────────────────────────────────────────────────
 * FILE   src/driver/input_parser.cu
 * ROLE   reads the candidate file into groups
 *
 * HOW    One equation per line, optionally prefixed with "group_id:". Lines
 *        sharing an id become one group; a line without an id is its own
 *        group. Blank lines and lines starting with # are skipped.
 *
 * NOTE   Grouping is what lets the driver stop testing a group as soon as
 *        one of its equations turns out composite.
 * ───────────────────────────────────────────────────────────────────────────── */
#include "driver/input_parser.h"
#include <fstream>
#include <sstream>
#include <string>
#include <map>
#include <stdexcept>

std::vector<GroupInfo> parse_input(const char *path)
{
    std::ifstream fin(path);
    if (!fin)
        throw std::runtime_error(std::string("Cannot open file: ") + path);

    std::vector<std::string> order;
    std::map<std::string, GroupInfo> groups;
    int auto_id = 0;

    std::string line;
    while (std::getline(fin, line))
    {
        while (!line.empty() && (line.back() == '\r' || line.back() == ' ' || line.back() == '\t'))
            line.pop_back();
        if (line.empty() || line[0] == '#')
            continue;

        std::string label, equation;
        auto colon = line.find(':');
        if (colon != std::string::npos)
        {
            label = line.substr(0, colon);
            equation = line.substr(colon + 1);
        }
        else
        {
            label = "__auto_" + std::to_string(auto_id++);
            equation = line;
        }

        auto trim = [](std::string &s)
        {
            while (!s.empty() && std::isspace((unsigned char)s.front()))
                s.erase(s.begin());
            while (!s.empty() && std::isspace((unsigned char)s.back()))
                s.pop_back();
        };
        trim(label);
        trim(equation);
        if (equation.empty())
            continue;

        if (groups.find(label) == groups.end())
        {
            order.push_back(label);
            groups[label].label = label;
        }
        groups[label].equations.push_back(equation);
    }

    std::vector<GroupInfo> result;
    result.reserve(order.size());
    for (auto &lbl : order)
        result.push_back(std::move(groups[lbl]));
    return result;
}
