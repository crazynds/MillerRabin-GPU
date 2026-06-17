#!/usr/bin/env bash
set -euo pipefail

BINARY="./build/bench_mr_gpu"
INPUT="example.txt"

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

case "${1:-}" in
    test)
        exec "$BINARY" --test
        ;;
    prof)
        exec "$BINARY" --bench-ops-long
        ;;
    "")
        if [[ ! -f "$INPUT" ]]; then
            echo "Input file not found: $INPUT" >&2
            exit 1
        fi
        exec "$BINARY" --progress --report "$INPUT"
        ;;
    *)
        echo "Usage: $0 [test|prof]" >&2
        exit 1
        ;;
esac
