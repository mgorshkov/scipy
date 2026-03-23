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

#pragma once

#include <cmath>

#if defined(__i386__) || defined(__x86_64__)
#include <immintrin.h>
#endif

// Fallback implementations for SVML intrinsics when not available
#if defined(__i386__) || defined(__x86_64__)
#if defined(ENABLE_AVX) || defined(ENABLE_AVX2) || defined(ENABLE_AVX512)
#ifndef _mm_log_pd
static inline __m128d _mm_log_pd(__m128d x) {
    alignas(16) double tmp[2];
    _mm_store_pd(tmp, x);
    tmp[0] = log(tmp[0]);
    tmp[1] = log(tmp[1]);
    return _mm_load_pd(tmp);
}
#endif

#ifndef _mm_exp_pd
static inline __m128d _mm_exp_pd(__m128d x) {
    alignas(16) double tmp[2];
    _mm_store_pd(tmp, x);
    tmp[0] = exp(tmp[0]);
    tmp[1] = exp(tmp[1]);
    return _mm_load_pd(tmp);
}
#endif

#if defined(ENABLE_AVX2) || defined(ENABLE_AVX512)
#ifndef _mm256_log_pd
static inline __m256d _mm256_log_pd(__m256d x) {
    alignas(32) double tmp[4];
    _mm256_store_pd(tmp, x);
    tmp[0] = log(tmp[0]);
    tmp[1] = log(tmp[1]);
    tmp[2] = log(tmp[2]);
    tmp[3] = log(tmp[3]);
    return _mm256_load_pd(tmp);
}
#endif

#ifndef _mm256_exp_pd
static inline __m256d _mm256_exp_pd(__m256d x) {
    alignas(32) double tmp[4];
    _mm256_store_pd(tmp, x);
    tmp[0] = exp(tmp[0]);
    tmp[1] = exp(tmp[1]);
    tmp[2] = exp(tmp[2]);
    tmp[3] = exp(tmp[3]);
    return _mm256_load_pd(tmp);
}
#endif
#endif//ENABLE_AVX2 || ENABLE_AVX512

#ifdef ENABLE_AVX512
#ifndef _mm512_log_pd
static inline __m512d _mm512_log_pd(__m512d x) {
    alignas(64) double tmp[8];
    _mm512_store_pd(tmp, x);
    for (int i = 0; i < 8; ++i) {
        tmp[i] = log(tmp[i]);
    }
    return _mm512_load_pd(tmp);
}
#endif

#ifndef _mm512_exp_pd
static inline __m512d _mm512_exp_pd(__m512d x) {
    alignas(64) double tmp[8];
    _mm512_store_pd(tmp, x);
    for (int i = 0; i < 8; ++i) {
        tmp[i] = exp(tmp[i]);
    }
    return _mm512_load_pd(tmp);
}
#endif//ENABLE_AVX512
#endif
#endif
#endif
