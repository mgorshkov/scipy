/*
⚡ SciPy methods in C++ | SIMD (AVX2/AVX512/AMX) CPU

Copyright (c) 2022-2026 Mikhail Gorshkov (mikhail.gorshkov@gmail.com)
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include <iostream>
#include <iomanip>
#include <string>
#include <cstring>
#include <cstdio>
#include <memory>
#include <array>
#include <functional>
#include <time.h>

#include <scipy/special/betainc.hpp>

typedef double (*IncompleteBetafunc)(double, double, double);

struct BenchmarkResult {
    double result;
    std::uint64_t time_ns;
    int loops;
};

BenchmarkResult measureIncompleteBetaFunction(IncompleteBetafunc func) {
    timespec start;
    clock_gettime(CLOCK_MONOTONIC, &start);

    np::float_ a = 0.5 * 99997;
    np::float_ b = 0.5 * 99997;

    np::float_ x = 0.4;
    int count = 0;
    np::float_ res = 0;

    while (x < 0.6) {
        ++count;
        res += func(a, b, x);
        x += 0.000001;
    }

    timespec stop;
    clock_gettime(CLOCK_MONOTONIC, &stop);

    std::uint64_t diff = 1000000000L * (stop.tv_sec - start.tv_sec) + stop.tv_nsec - start.tv_nsec;

    return {res, diff, count};
}

// Execute a command and capture its stdout
std::string execCommand(const char* cmd) {
    std::array<char, 256> buffer;
    std::string result;
    auto pipe = std::unique_ptr<FILE, decltype([](FILE* f) { pclose(f); })>(popen(cmd, "r"));
    if (!pipe) {
        return "ERROR: failed to run command";
    }
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    return result;
}

// Parse a line like "Time = 123456789 ns" from Python output
std::uint64_t parseTimeNs(const std::string& output) {
    auto pos = output.find("Time = ");
    if (pos == std::string::npos) return 0;
    pos += 7; // skip "Time = "
    auto end = output.find(" ns", pos);
    if (end == std::string::npos) return 0;
    return std::stoull(output.substr(pos, end - pos));
}

// Parse "Result = <value>" from Python output
double parseResult(const std::string& output) {
    auto pos = output.find("Result = ");
    if (pos == std::string::npos) return 0;
    pos += 9; // skip "Result = "
    auto end = output.find('\n', pos);
    if (end == std::string::npos) return 0;
    return std::stod(output.substr(pos, end - pos));
}

// Parse "Loops = <count>" from Python output
int parseLoops(const std::string& output) {
    auto pos = output.find("Loops = ");
    if (pos == std::string::npos) return 0;
    pos += 8; // skip "Loops = "
    auto end = output.find('\n', pos);
    if (end == std::string::npos) return 0;
    return std::stoi(output.substr(pos, end - pos));
}

int main(int, char**) {
    std::cout << "============================================================" << std::endl;
    std::cout << "  Performance Comparison: C++ scipy vs Python scipy betainc" << std::endl;
    std::cout << "  AVX2 Optimization Enabled" << std::endl;
    std::cout << "============================================================" << std::endl;
    std::cout << std::endl;
    std::cout << "Test parameters:" << std::endl;
    std::cout << "  a = 0.5 * 99997" << std::endl;
    std::cout << "  b = 0.5 * 99997" << std::endl;
    std::cout << "  x range: 0.4 to 0.6 (step 0.000001)" << std::endl;
    std::cout << std::endl;

    // --- C++ scipy benchmark ---
    std::cout << "--- C++ scipy (AVX2) ---" << std::endl;
    auto cppResult = measureIncompleteBetaFunction(scipy::special::betainc);
    std::cout << "Result = " << cppResult.result << std::endl;
    std::cout << "Time = " << cppResult.time_ns << " ns" << std::endl;
    std::cout << "Loops = " << cppResult.loops << std::endl;
    std::cout << std::endl;

    // --- Python scipy benchmark ---
    std::cout << "--- Python scipy ---" << std::endl;
    std::cout << "Running Python benchmark via subprocess..." << std::endl;

    // Get the directory of this executable to find the python script
    std::string pythonCmd = "python3 ";
#ifdef PYTHON_BENCHMARK_DIR
    pythonCmd += PYTHON_BENCHMARK_DIR;
#else
    pythonCmd += ".";
#endif
    pythonCmd += "/python_benchmark.py";

    std::string pyOutput = execCommand(pythonCmd.c_str());

    if (pyOutput.find("ERROR") != std::string::npos) {
        std::cerr << "Failed to run Python benchmark: " << pyOutput << std::endl;
        std::cerr << "Make sure Python 3 and scipy are installed." << std::endl;
        return 1;
    }

    std::cout << pyOutput;
    std::cout << std::endl;

    auto pyTimeNs = parseTimeNs(pyOutput);
    auto pyResult = parseResult(pyOutput);
    auto pyLoops = parseLoops(pyOutput);

    // --- Comparison summary ---
    std::cout << "============================================================" << std::endl;
    std::cout << "  Comparison Summary" << std::endl;
    std::cout << "============================================================" << std::endl;
    std::cout << std::left << std::setw(25) << "Implementation"
              << std::right << std::setw(18) << "Time (ns)"
              << std::setw(14) << "Loops"
              << std::setw(20) << "Speedup vs Python" << std::endl;
    std::cout << std::string(77, '-') << std::endl;

    auto printRow = [&](const std::string& name, std::uint64_t timeNs, int loops) {
        double speedup = static_cast<double>(pyTimeNs) / static_cast<double>(timeNs);
        std::cout << std::left << std::setw(25) << name
                  << std::right << std::setw(18) << timeNs
                  << std::setw(14) << loops
                  << std::setw(20) << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;
    };

    printRow("C++ scipy (AVX2)", cppResult.time_ns, cppResult.loops);
    printRow("Python scipy", pyTimeNs, pyLoops);

    std::cout << std::string(77, '-') << std::endl;
    std::cout << std::endl;

    // Verify results match
    const double tolerance = 1e-10;
    double cppVsPy = std::abs(cppResult.result - pyResult);
    std::cout << "Result verification:" << std::endl;
    std::cout << "  C++ scipy result:  " << std::setprecision(15) << cppResult.result << std::endl;
    std::cout << "  Python scipy result: " << std::setprecision(15) << pyResult << std::endl;
    std::cout << "  Absolute difference: " << cppVsPy << std::endl;
    if (cppVsPy < tolerance) {
        std::cout << "  ✓ Results match within tolerance." << std::endl;
    } else {
        std::cout << "  ⚠ Results differ! (tolerance = " << tolerance << ")" << std::endl;
    }

    return 0;
}
