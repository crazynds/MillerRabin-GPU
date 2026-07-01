#!/usr/bin/env bash
set -euo pipefail

BINARY="./build/bench_mr_gpu"
INPUT="example.txt"
INPUT_BIG="example_big"
INPUT_SMALL="example_small.txt"

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Parse flags: --big and --small can appear anywhere in the args
BIG=0
SMALL=0
ARGS=()
for arg in "$@"; do
    if [[ "$arg" == "--big-1" ]]; then
        BIG=1
    elif [[ "$arg" == "--big-2" ]]; then
        BIG=2
    elif [[ "$arg" == "--big-3" ]]; then
        BIG=3
    elif [[ "$arg" == "--small" ]]; then
        SMALL=1
    else
        ARGS+=("$arg")
    fi
done

MODE="${ARGS[0]:-}"
THREADS="${ARGS[1]:-}"

if [[ "$BIG" == "1" ]]; then
    ACTIVE_INPUT="${INPUT_BIG}_1.txt"
elif [[ "$BIG" == "2" ]]; then
    ACTIVE_INPUT="${INPUT_BIG}_2.txt"
elif [[ "$BIG" == "3" ]]; then
    ACTIVE_INPUT="${INPUT_BIG}_3.txt"
elif [[ "$SMALL" == "1" ]]; then
    ACTIVE_INPUT="$INPUT_SMALL"
else
    ACTIVE_INPUT="$INPUT"
fi

case "$MODE" in
    test)
        exec "$BINARY" --test
        ;;
    prof)
        exec "$BINARY" --bench-ops-long
        ;;
    cpu)
        if [[ ! -f "$ACTIVE_INPUT" ]]; then echo "Input file not found: $ACTIVE_INPUT" >&2; exit 1; fi
        exec "$BINARY" --cpu --progress --report "$ACTIVE_INPUT"
        ;;
    cpu-parallel)
        if [[ ! -f "$ACTIVE_INPUT" ]]; then echo "Input file not found: $ACTIVE_INPUT" >&2; exit 1; fi
        if [[ -n "$THREADS" ]]; then
            exec "$BINARY" --threads "$THREADS" --progress --report "$ACTIVE_INPUT"
        else
            exec "$BINARY" --cpu-parallel --progress --report "$ACTIVE_INPUT"
        fi
        ;;
    bench)
        if [[ ! -f "$ACTIVE_INPUT" ]]; then echo "Input file not found: $ACTIVE_INPUT" >&2; exit 1; fi
        exec "$BINARY" --progress "$ACTIVE_INPUT"
        ;;
    "")
        if [[ ! -f "$ACTIVE_INPUT" ]]; then echo "Input file not found: $ACTIVE_INPUT" >&2; exit 1; fi
        exec "$BINARY" --progress --report "$ACTIVE_INPUT"
        ;;
    *)
        echo "Usage: $0 [--big|--small] [test|prof|bench|cpu|cpu-parallel [N_THREADS]]" >&2
        exit 1
        ;;
esac
