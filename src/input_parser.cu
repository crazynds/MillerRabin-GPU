#include "input_parser.h"
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

    // Preserve insertion order while grouping.
    std::vector<std::string> order;           // ordered unique labels
    std::map<std::string, GroupInfo> groups;  // label → group
    int auto_id = 0;

    std::string line;
    while (std::getline(fin, line))
    {
        // Strip trailing whitespace
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

        // Trim leading/trailing whitespace from both parts
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
