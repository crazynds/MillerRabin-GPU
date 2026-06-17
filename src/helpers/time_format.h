#pragma once
// helpers/time_format.h — time formatting that automatically chooses the best unit
// (min / s / ms / µs / ns). Used by all the project's time prints.

#include <string>
#include <cstdio>
#include <cmath>

// Formats an interval given in SECONDS, choosing the most readable unit.
// Switches to the smaller unit only when the value falls below 0.1 of it — so
// 0.6 ms shows as "0.600 ms" (and not "600.000 us").
//   >= 60 s    → minutes
//   >= 1 s     → seconds
//   >= 0.1 ms  → milliseconds
//   >= 0.1 µs  → microseconds
//   otherwise  → nanoseconds
inline std::string fmt_time(double seconds)
{
    char buf[32];
    double a = std::fabs(seconds);
    if (a >= 60.0)
        snprintf(buf, sizeof(buf), "%.2f min", seconds / 60.0);
    else if (a >= 1.0)
        snprintf(buf, sizeof(buf), "%.3f s", seconds);
    else if (a >= 1e-4)
        snprintf(buf, sizeof(buf), "%.3f ms", seconds * 1e3);
    else if (a >= 1e-7)
        snprintf(buf, sizeof(buf), "%.3f us", seconds * 1e6);
    else
        snprintf(buf, sizeof(buf), "%.0f ns", seconds * 1e9);
    return std::string(buf);
}

// Convenience for times already measured in MILLISECONDS.
inline std::string fmt_time_ms(double milliseconds)
{
    return fmt_time(milliseconds / 1e3);
}
