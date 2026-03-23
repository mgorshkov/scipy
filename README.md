[![Build status](https://ci.appveyor.com/api/projects/status/bdews6m7dh2botlx/branch/main?svg=true)](https://ci.appveyor.com/project/mgorshkov/scipy/branch/main)

![np logo](doc/logo.svg)

# About
⚡ SciPy methods in C++ | CUDA GPU + SIMD (AVX2/AVX512/AMX) CPU

# Requirements
C++20-compatible compiler:
* gcc 13 or higher
* clang 14 or higher
* Visual Studio 2019 or higher
* CUDA development environment (NVIDIA CUDA Toolkit, and compatible NVIDIA drivers installed) to use CUDA optimizations (nvcc 12 or higher)

# Repo
```
git clone https://github.com/mgorshkov/scipy.git
```

# Build library and unit tests
```
./scripts/build.sh
```

# Build docs
```
cmake --build . --target doc
```

Open scipy/build/doc/html/index.html in your browser.

# Install
```
cmake .. -DCMAKE_INSTALL_PREFIX:PATH=~/scipy_install
cmake --build . --target install
```

# Usage example (samples/stats)
```
#include <iostream>

#include <np/Creators.hpp>
#include <scipy/stats/mode.hpp>

int main(int, char **) {
    using namespace np;
    using namespace scipy;

    // Mode calculation
    Size size = 10000000;
    auto r = random::rand(size);
    auto m = stats::mode(r);
    std::cout << "mode=" << m.first << " " << m.second;
    return 0;
}
```
## How to build the sample

1. Clone the repo
```
git clone https://github.com/mgorshkov/scipy.git
```
2. cd samples/stats
```
cd samples/stats
```
3. Make build dir
```
mkdir -p build && cd build
```
4. Configure cmake
```
cmake -DCMAKE_BUILD_TYPE=Release ..
```
5. Build
### Linux/MacOS
```
cmake --build .
```
### Windows
```
cmake --build . --config Release
```
6. Run the app
```
$./stats

```

# C++ vs Python scipy betainc function performance comparison

## How to build the benchmark

1. Clone the repo
```
git clone https://github.com/mgorshkov/scipy.git
```
2. cd benchmarks/betainc
```
cd benchmarks/betainc
```
3. Make build dir
```
mkdir -p build && cd build
```
4. Configure cmake
```
cmake -DCMAKE_BUILD_TYPE=Release ..
```
5. Build
### Linux/MacOS
```
cmake --build .
```
### Windows
```
cmake --build . --config Release
```
6. Run Results
```
./betainc_comparison

============================================================
  Performance Comparison: C++ scipy vs Python scipy betainc
  AVX2 Optimization Enabled
============================================================

Test parameters:
  a = 0.5 * 99997
  b = 0.5 * 99997
  x range: 0.4 to 0.6 (step 0.000001)

--- C++ scipy (AVX2) ---
Result = 99999.5
Time = 115920731 ns
Loops = 200000

--- Python scipy ---
Running Python benchmark via subprocess...
Result = 99999.49999735961
Time = 258229902 ns
Loops = 200000

============================================================
  Comparison Summary
============================================================
Implementation                    Time (ns)         Loops   Speedup vs Python
-----------------------------------------------------------------------------
C++ scipy (AVX2)                  115920731        200000                2.23x
Python scipy                      258229902        200000                1.00x
-----------------------------------------------------------------------------

Result verification:
  C++ scipy result:  99999.499997359671397
  Python scipy result: 99999.499997359613189
  Absolute difference: 0.000000000058208
  ✓ Results match within tolerance.
```

# Links
* ⚡ NumPy-style arrays in C++ | CUDA GPU + SIMD (AVX2/AVX512/AMX) CPU: https://github.com/mgorshkov/np
* ⚡ Data manipulation and analysis library in C++ | CUDA GPU + SIMD (AVX2/AVX512/AMX) CPU: https://github.com/mgorshkov/pd
* ⚡ ML methods in C++ | CUDA GPU + SIMD (AVX2/AVX512/AMX) CPU: https://github.com/mgorshkov/sklearn
