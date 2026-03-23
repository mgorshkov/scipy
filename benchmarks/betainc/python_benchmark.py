#!/usr/bin/env python3
"""
Python scipy betainc benchmark - called by the C++ comparison benchmark.
Uses the same test parameters as the C++ benchmark for fair comparison.
"""
import time
import sys
import scipy.special


def benchmark_python_scipy():
    a = 0.5 * 99997
    b = 0.5 * 99997
    x = 0.4
    count = 0
    res = 0.0

    start = time.perf_counter_ns()

    while x < 0.6:
        count += 1
        res += scipy.special.betainc(a, b, x)
        x += 0.000001

    stop = time.perf_counter_ns()

    diff = stop - start
    print(f"Result = {res}")
    print(f"Time = {diff} ns")
    print(f"Loops = {count}")


if __name__ == "__main__":
    benchmark_python_scipy()
