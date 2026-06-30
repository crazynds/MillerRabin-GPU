#!/usr/bin/env bash
set -euo pipefail

BINARY="./build/bench_mr_gpu"
INPUT="example.txt"

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

MODE="${1:-}"
THREADS="${2:-}"   # optional: number of CPU threads (e.g. ./run.sh cpu-parallel 8)

case "$MODE" in
    test)
        exec "$BINARY" --test
        ;;
    prof)
        exec "$BINARY" --bench-ops-long
        ;;
    cpu)
        if [[ ! -f "$INPUT" ]]; then echo "Input file not found: $INPUT" >&2; exit 1; fi
        exec "$BINARY" --cpu --progress --report "$INPUT"
        ;;
    cpu-parallel)
        if [[ ! -f "$INPUT" ]]; then echo "Input file not found: $INPUT" >&2; exit 1; fi
        if [[ -n "$THREADS" ]]; then
            exec "$BINARY" --threads "$THREADS" --progress --report "$INPUT"
        else
            exec "$BINARY" --cpu-parallel --progress --report "$INPUT"
        fi
        ;;
    "")
        if [[ ! -f "$INPUT" ]]; then echo "Input file not found: $INPUT" >&2; exit 1; fi
        exec "$BINARY" --progress --report "$INPUT"
        ;;
    *)
        echo "Usage: $0 [test|prof|cpu|cpu-parallel [N_THREADS]]" >&2
        exit 1
        ;;
esac
