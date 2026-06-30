#!/usr/bin/env bash
set -euo pipefail

BINARY="./build/bench_mr_gpu"
INPUT="example.txt"
INPUT_BIG="example_big.txt"
INPUT_SMALL="example_small.txt"

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Parse flags: --big and --small can appear anywhere in the args
BIG=0
SMALL=0
ARGS=()
for arg in "$@"; do
    if [[ "$arg" == "--big" ]]; then
        BIG=1
    elif [[ "$arg" == "--small" ]]; then
        SMALL=1
    else
        ARGS+=("$arg")
    fi
done

MODE="${ARGS[0]:-}"
THREADS="${ARGS[1]:-}"

if [[ "$BIG" == "1" ]]; then
    ACTIVE_INPUT="$INPUT_BIG"
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
    "")
        if [[ ! -f "$ACTIVE_INPUT" ]]; then echo "Input file not found: $ACTIVE_INPUT" >&2; exit 1; fi
        exec "$BINARY" --progress --report "$ACTIVE_INPUT"
        ;;
    *)
        echo "Usage: $0 [--big|--small] [test|prof|cpu|cpu-parallel [N_THREADS]]" >&2
        exit 1
        ;;
esac
