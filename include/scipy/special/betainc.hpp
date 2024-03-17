/*
Scientific methods on top of NP library

Copyright (c) 2023 Mikhail Gorshkov (mikhail.gorshkov@gmail.com)
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

#include <np/Array.hpp>

namespace scipy {
    namespace special {
#if defined(ENABLE_AVX2) || defined(ENABLE_AVX512)
        np::float_ horizontalAdd(const __m128d &a) {
            auto tmp = _mm_hadd_pd(a, a);
            double res;
            _mm_store_sd(&res, tmp);
            return res;
        }

        np::float_ horizontalAdd(const __m256d &a) {
            auto tmp1 = _mm256_hadd_pd(a, a);
            auto tmp2 = _mm256_extractf128_pd(tmp1, 1);
            auto tmp3 = _mm_add_sd(_mm256_castpd256_pd128(tmp1), tmp2);
            return _mm_cvtsd_f64(tmp3);
        }
#ifdef ENABLE_AVX512
        np::float_ horizontalAdd(const __m512d &a) {
            auto low = _mm512_castpd512_pd256(a);
            auto high = _mm512_extractf64x4_pd(a, 1);

            return horizontalAdd(low) + horizontalAdd(high);
        }
#endif
#endif
        np::float_ lbeta(np::float_ a, np::float_ b) {
            static np::float_ cof[] __attribute__((aligned(64))) = {57.1562356658629235, -59.5979603554754912,
                                                                    14.1360979747417471, -0.491913816097620199, 0.339946499848118887e-4,
                                                                    0.465236289270485756e-4, -0.983744753048795646e-4, 0.158088703224912494e-3,
                                                                    -0.210264441724104883e-3, 0.217439618115212643e-3, -0.164318106536763890e-3,
                                                                    0.844182239838527433e-4, -0.261908384015814087e-4, 0.368991826595316234e-5, 0.0, 0.0};
            if (a <= 0.0 || b <= 0.0) {
                throw std::runtime_error("a and b must be > 0");
            }
            np::float_ res = 0.0;
            constexpr np::float_ const1 = 5.24218750000000000;// 671/128
            constexpr np::float_ const2 = 0.5;
            constexpr np::float_ const3 = 0.999999999999997092;
            constexpr np::float_ const4 = 2.5066282746310005;
#ifdef ENABLE_AVX2
            np::float_ y[] = {a, b, a + b, 0};
            auto regy = _mm256_load_pd(&y[0]);
            auto regx = regy;

            auto regconst1 = _mm256_broadcast_sd(&const1);
            auto regtmp = _mm256_add_pd(regx, regconst1);

            auto regconst2 = _mm256_broadcast_sd(&const2);
            regtmp = _mm256_sub_pd(_mm256_mul_pd(_mm256_add_pd(regx, regconst2), _mm256_log_pd(regtmp)), regtmp);

            auto regcof0 = _mm256_load_pd(&cof[0]);
            auto regcof1 = _mm256_load_pd(&cof[4]);
            auto regcof2 = _mm256_load_pd(&cof[8]);
            auto regcof3 = _mm256_load_pd(&cof[12]);

            constexpr np::float_ const5 = 4.0;
            auto regconst5 = _mm256_broadcast_sd(&const5);

            static np::float_ add[] __attribute__((aligned(64))) = {1.0, 2.0, 3.0, 4.0};

            auto rega = _mm256_broadcast_sd(&y[0]);
            auto regb = _mm256_broadcast_sd(&y[1]);
            auto regab = _mm256_broadcast_sd(&y[2]);

            auto regadd = _mm256_load_pd(&add[0]);
            rega = _mm256_add_pd(rega, regadd);
            regb = _mm256_add_pd(regb, regadd);
            regab = _mm256_add_pd(regab, regadd);

            static np::float_ ser[4] __attribute__((aligned(64))) = {const3};

            auto regsera = _mm256_load_pd(&ser[0]);
            auto regserb = _mm256_load_pd(&ser[0]);
            auto regserab = _mm256_load_pd(&ser[0]);

            regsera = _mm256_add_pd(regsera, _mm256_div_pd(regcof0, rega));
            regserb = _mm256_add_pd(regserb, _mm256_div_pd(regcof0, regb));
            regserab = _mm256_add_pd(regserab, _mm256_div_pd(regcof0, regab));

            rega = _mm256_add_pd(rega, regconst5);
            regb = _mm256_add_pd(regb, regconst5);
            regab = _mm256_add_pd(regab, regconst5);

            regsera = _mm256_add_pd(regsera, _mm256_div_pd(regcof1, rega));
            regserb = _mm256_add_pd(regserb, _mm256_div_pd(regcof1, regb));
            regserab = _mm256_add_pd(regserab, _mm256_div_pd(regcof1, regab));

            rega = _mm256_add_pd(rega, regconst5);
            regb = _mm256_add_pd(regb, regconst5);
            regab = _mm256_add_pd(regab, regconst5);

            regsera = _mm256_add_pd(regsera, _mm256_div_pd(regcof2, rega));
            regserb = _mm256_add_pd(regserb, _mm256_div_pd(regcof2, regb));
            regserab = _mm256_add_pd(regserab, _mm256_div_pd(regcof2, regab));

            rega = _mm256_add_pd(rega, regconst5);
            regb = _mm256_add_pd(regb, regconst5);
            regab = _mm256_add_pd(regab, regconst5);

            regsera = _mm256_add_pd(regsera, _mm256_div_pd(regcof3, rega));
            regserb = _mm256_add_pd(regserb, _mm256_div_pd(regcof3, regb));
            regserab = _mm256_add_pd(regserab, _mm256_div_pd(regcof3, regab));

            np::float_ tmp[4] = {horizontalAdd(regsera),
                                 horizontalAdd(regserb),
                                 horizontalAdd(regserab)};
            auto regser = _mm256_load_pd(&tmp[0]);
            auto regconst4 = _mm256_broadcast_sd(&const4);

            regser = _mm256_mul_pd(regser, regconst4);
            regser = _mm256_div_pd(regser, regx);
            regser = _mm256_log_pd(regser);

            res += regtmp[0];
            res += regtmp[1];
            res -= regtmp[2];

            res += regser[0];
            res += regser[1];
            res -= regser[2];
#elif defined(ENABLE_AVX512)
            np::float_ y[] = {a, b, a + b, 0};
            auto regy = _mm256_load_pd(&y[0]);
            auto regx = regy;

            auto regconst1 = _mm256_broadcast_sd(&const1);
            auto regtmp = _mm256_add_pd(regx, regconst1);

            auto regconst2 = _mm256_broadcast_sd(&const2);
            regtmp = _mm256_sub_pd(_mm256_mul_pd(_mm256_add_pd(regx, regconst2), _mm256_log_pd(regtmp)), regtmp);

            auto regcof0 = _mm512_load_pd(&cof[0]);
            auto regcof1 = _mm512_load_pd(&cof[8]);

            constexpr np::float_ const5 = 8.0;
            auto regconst5 = _mm512_set1_pd(const5);

            static np::float_ add[] __attribute__((aligned(64))) = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};

            auto rega = _mm512_set1_pd(y[0]);
            auto regb = _mm512_set1_pd(y[1]);
            auto regab = _mm512_set1_pd(y[2]);

            auto regadd = _mm512_load_pd(&add[0]);
            rega = _mm512_add_pd(rega, regadd);
            regb = _mm512_add_pd(regb, regadd);
            regab = _mm512_add_pd(regab, regadd);

            static np::float_ ser[8] __attribute__((aligned(64))) = {const3};

            auto regsera = _mm512_load_pd(&ser[0]);
            auto regserb = _mm512_load_pd(&ser[0]);
            auto regserab = _mm512_load_pd(&ser[0]);

            regsera = _mm512_add_pd(regsera, _mm512_div_pd(regcof0, rega));
            regserb = _mm512_add_pd(regserb, _mm512_div_pd(regcof0, regb));
            regserab = _mm512_add_pd(regserab, _mm512_div_pd(regcof0, regab));

            rega = _mm512_add_pd(rega, regconst5);
            regb = _mm512_add_pd(regb, regconst5);
            regab = _mm512_add_pd(regab, regconst5);

            regsera = _mm512_add_pd(regsera, _mm512_div_pd(regcof1, rega));
            regserb = _mm512_add_pd(regserb, _mm512_div_pd(regcof1, regb));
            regserab = _mm512_add_pd(regserab, _mm512_div_pd(regcof1, regab));

            np::float_ tmp[4] = {horizontalAdd(regsera),
                                 horizontalAdd(regserb),
                                 horizontalAdd(regserab)};
            auto regser = _mm256_load_pd(&tmp[0]);
            auto regconst4 = _mm256_broadcast_sd(&const4);

            regser = _mm256_mul_pd(regser, regconst4);
            regser = _mm256_div_pd(regser, regx);
            regser = _mm256_log_pd(regser);

            res += regtmp[0];
            res += regtmp[1];
            res -= regtmp[2];

            res += regser[0];
            res += regser[1];
            res -= regser[2];
#else
            {
                np::float_ y = a;
                np::float_ x = y;
                np::float_ tmp = x + const1;
                res = (x + const2) * std::log(tmp) - tmp;
                np::float_ ser = const3;
                for (auto c: cof) {
                    ser += c / ++y;
                }
                res += std::log(const4 * ser / x);
            }
            {
                np::float_ y = b;
                np::float_ x = y;
                np::float_ tmp = x + const1;
                res += (x + const2) * std::log(tmp) - tmp;
                np::float_ ser = const3;
                for (auto c: cof) {
                    ser += c / ++y;
                }
                res += std::log(const4 * ser / x);
            }
            {
                np::float_ y = a + b;
                np::float_ x = y;
                np::float_ tmp = x + const1;
                res -= (x + const2) * std::log(tmp) - tmp;
                np::float_ ser = const3;
                for (auto c: cof) {
                    ser += c / ++y;
                }
                res -= std::log(const4 * ser / x);
            }
#endif
            return res;
        }

        // Modified Lentz's method
        np::float_ fr(np::float_ a, np::float_ b, np::float_ x) {
            np::float_ c = 1.0;
            np::float_ d = 1.0 / (1.0 - (a + b) * x / (a + 1));
            np::float_ h = d;

            static const int maxIter = 5000;

            for (int m = 1; m < maxIter; ++m) {
                auto aa = m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m));
                c = 1.0 + aa / c;
                d = 1.0 / (1.0 + aa * d);
                h *= d * c;

                aa = -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1));
                c = 1.0 + aa / c;
                d = 1.0 / (1.0 + aa * d);
                auto delta = d * c;
                h *= delta;

                if (std::fabs(delta - 1.0) <= std::numeric_limits<np::float_>::epsilon()) {
                    break;
                }
            }

            return h;
        }

        np::float_ fract(np::float_ a, np::float_ b, np::float_ x) {
            auto res = std::exp(a * std::log(x) + b * std::log(1.0 - x) - lbeta(a, b));

            return x < (a + 1.0) / (a + b + 2.0) ? res * fr(a, b, x) / a : 1.0 - res * fr(b, a, 1.0 - x) / b;
        }

        np::float_ gaussLegendreQuadrature(np::float_ a, np::float_ b, np::float_ x) {
            static np::float_ y[] __attribute__((aligned(64))) = {0.0021695375159141994,
                                                                  0.011413521097787704, 0.027972308950302116, 0.051727015600492421,
                                                                  0.082502225484340941, 0.12007019910960293, 0.16415283300752470,
                                                                  0.21442376986779355, 0.27051082840644336, 0.33199876341447887,
                                                                  0.39843234186401943, 0.46931971407375483, 0.54413605556657973,
                                                                  0.62232745288031077, 0.70331500465597174, 0.78649910768313447,
                                                                  0.87126389619061517, 0.95698180152629142};
            static np::float_ w[] __attribute__((aligned(64))) = {0.0055657196642445571,
                                                                  0.012915947284065419, 0.020181515297735382, 0.027298621498568734,
                                                                  0.034213810770299537, 0.040875750923643261, 0.047235083490265582,
                                                                  0.053244713977759692, 0.058860144245324798, 0.064039797355015485,
                                                                  0.068745323835736408, 0.072941885005653087, 0.076598410645870640,
                                                                  0.079687828912071670, 0.082187266704339706, 0.084078218979661945,
                                                                  0.085346685739338721, 0.085983275670394821};

            constexpr np::float_ zero = 0.0;
            constexpr np::float_ one = 1.0;

            np::float_ mu = a / (a + b);
            np::float_ lnmu = std::log(mu);
            np::float_ muc = one - mu;
            np::float_ lnmuc = std::log(muc);
            np::float_ t = std::sqrt(a * b / ((a + b) * (a + b) * (a + b + one)));
            np::float_ xu;
            if (x > mu) {
                xu = std::min(one, std::max(mu + 10.0 * t, x + 5.0 * t));
            } else {
                xu = std::max(zero, std::min(mu - 10.0 * t, x - 5.0 * t));
            }
            np::float_ deltax = xu - x;

            np::float_ a1 = a - one;
            np::float_ b1 = b - one;
            np::float_ sum = zero;
#ifdef ENABLE_AVX2
            auto regx = _mm256_broadcast_sd(&x);
            auto regone = _mm256_broadcast_sd(&one);
            auto reglnmu = _mm256_broadcast_sd(&lnmu);
            auto reglnmuc = _mm256_broadcast_sd(&lnmuc);

            auto rega1 = _mm256_broadcast_sd(&a1);
            auto regb1 = _mm256_broadcast_sd(&b1);

            auto reg0 = _mm256_load_pd(&y[0]);
            auto reg1 = _mm256_load_pd(&y[4]);
            auto reg2 = _mm256_load_pd(&y[8]);
            auto reg3 = _mm256_load_pd(&y[12]);
            auto reg4 = _mm_load_pd(&y[16]);

            auto regdeltax = _mm256_broadcast_sd(&deltax);
            reg0 = _mm256_mul_pd(reg0, regdeltax);
            reg1 = _mm256_mul_pd(reg1, regdeltax);
            reg2 = _mm256_mul_pd(reg2, regdeltax);
            reg3 = _mm256_mul_pd(reg3, regdeltax);
            reg4 = _mm_mul_pd(reg4, _mm_set1_pd(deltax));

            reg0 = _mm256_add_pd(regx, reg0);
            reg1 = _mm256_add_pd(regx, reg1);
            reg2 = _mm256_add_pd(regx, reg2);
            reg3 = _mm256_add_pd(regx, reg3);
            reg4 = _mm_add_pd(_mm_set1_pd(x), reg4);

            auto reg10 = _mm256_sub_pd(regone, reg0);
            auto reg11 = _mm256_sub_pd(regone, reg1);
            auto reg12 = _mm256_sub_pd(regone, reg2);
            auto reg13 = _mm256_sub_pd(regone, reg3);
            auto reg14 = _mm_sub_pd(_mm_set1_pd(one), reg4);

            reg0 = _mm256_log_pd(reg0);
            reg1 = _mm256_log_pd(reg1);
            reg2 = _mm256_log_pd(reg2);
            reg3 = _mm256_log_pd(reg3);
            reg4 = _mm_log_pd(reg4);

            reg0 = _mm256_sub_pd(reg0, reglnmu);
            reg1 = _mm256_sub_pd(reg1, reglnmu);
            reg2 = _mm256_sub_pd(reg2, reglnmu);
            reg3 = _mm256_sub_pd(reg3, reglnmu);
            reg4 = _mm_sub_pd(reg4, _mm_set1_pd(lnmu));

            reg0 = _mm256_mul_pd(reg0, rega1);
            reg1 = _mm256_mul_pd(reg1, rega1);
            reg2 = _mm256_mul_pd(reg2, rega1);
            reg3 = _mm256_mul_pd(reg3, rega1);
            reg4 = _mm_mul_pd(reg4, _mm_set1_pd(a1));

            reg10 = _mm256_log_pd(reg10);
            reg11 = _mm256_log_pd(reg11);
            reg12 = _mm256_log_pd(reg12);
            reg13 = _mm256_log_pd(reg13);
            reg14 = _mm_log_pd(reg14);

            reg10 = _mm256_sub_pd(reg10, reglnmuc);
            reg11 = _mm256_sub_pd(reg11, reglnmuc);
            reg12 = _mm256_sub_pd(reg12, reglnmuc);
            reg13 = _mm256_sub_pd(reg13, reglnmuc);
            reg14 = _mm_sub_pd(reg14, _mm_set1_pd(lnmuc));

            reg10 = _mm256_mul_pd(reg10, regb1);
            reg11 = _mm256_mul_pd(reg11, regb1);
            reg12 = _mm256_mul_pd(reg12, regb1);
            reg13 = _mm256_mul_pd(reg13, regb1);
            reg14 = _mm_mul_pd(reg14, _mm_set1_pd(b1));

            reg0 = _mm256_add_pd(reg0, reg10);
            reg1 = _mm256_add_pd(reg1, reg11);
            reg2 = _mm256_add_pd(reg2, reg12);
            reg3 = _mm256_add_pd(reg3, reg13);
            reg4 = _mm_add_pd(reg4, reg14);

            reg0 = _mm256_exp_pd(reg0);
            reg1 = _mm256_exp_pd(reg1);
            reg2 = _mm256_exp_pd(reg2);
            reg3 = _mm256_exp_pd(reg3);
            reg4 = _mm_exp_pd(reg4);

            auto regw0 = _mm256_load_pd(&w[0]);
            auto regw1 = _mm256_load_pd(&w[4]);
            auto regw2 = _mm256_load_pd(&w[8]);
            auto regw3 = _mm256_load_pd(&w[12]);
            auto regw4 = _mm_load_pd(&w[16]);

            reg0 = _mm256_mul_pd(regw0, reg0);
            reg1 = _mm256_mul_pd(regw1, reg1);
            reg2 = _mm256_mul_pd(regw2, reg2);
            reg3 = _mm256_mul_pd(regw3, reg3);
            reg4 = _mm_mul_pd(regw4, reg4);

            reg0 = _mm256_add_pd(reg0, reg1);
            reg0 = _mm256_add_pd(reg0, reg2);
            reg0 = _mm256_add_pd(reg0, reg3);
            reg0 = _mm256_add_pd(reg0, _mm256_castpd128_pd256(reg4));

            sum = horizontalAdd(reg0);
#elif defined(ENABLE_AVX512)
            auto regdeltax = _mm512_set1_pd(deltax);
            auto regx = _mm512_set1_pd(x);
            auto regone = _mm512_set1_pd(one);
            auto reglnmu = _mm512_set1_pd(lnmu);
            auto reglnmuc = _mm512_set1_pd(lnmuc);

            auto rega1 = _mm512_set1_pd(a1);
            auto regb1 = _mm512_set1_pd(b1);

            auto reg0 = _mm512_load_pd(&y[0]);
            auto reg1 = _mm512_load_pd(&y[8]);
            auto reg2 = _mm_load_pd(&y[16]);

            reg0 = _mm512_mul_pd(reg0, regdeltax);
            reg1 = _mm512_mul_pd(reg1, regdeltax);
            reg2 = _mm_mul_pd(reg2, _mm_set1_pd(deltax));

            reg0 = _mm512_add_pd(regx, reg0);
            reg1 = _mm512_add_pd(regx, reg1);
            reg2 = _mm_add_pd(_mm_set1_pd(x), reg2);

            auto reg10 = _mm512_sub_pd(regone, reg0);
            auto reg11 = _mm512_sub_pd(regone, reg1);
            auto reg12 = _mm_sub_pd(_mm_set1_pd(one), reg2);

            reg0 = _mm512_log_pd(reg0);
            reg1 = _mm512_log_pd(reg1);
            reg2 = _mm_log_pd(reg2);

            reg0 = _mm512_sub_pd(reg0, reglnmu);
            reg1 = _mm512_sub_pd(reg1, reglnmu);
            reg2 = _mm_sub_pd(reg2, _mm_set1_pd(lnmu));

            reg0 = _mm512_mul_pd(reg0, rega1);
            reg1 = _mm512_mul_pd(reg1, rega1);
            reg2 = _mm_mul_pd(reg2, _mm_set1_pd(a1));

            reg10 = _mm512_log_pd(reg10);
            reg11 = _mm512_log_pd(reg11);
            reg12 = _mm_log_pd(reg12);

            reg10 = _mm512_sub_pd(reg10, reglnmuc);
            reg11 = _mm512_sub_pd(reg11, reglnmuc);
            reg12 = _mm_sub_pd(reg12, _mm_set1_pd(lnmuc));

            reg10 = _mm512_mul_pd(reg10, regb1);
            reg11 = _mm512_mul_pd(reg11, regb1);
            reg12 = _mm_mul_pd(reg12, _mm_set1_pd(b1));

            reg0 = _mm512_add_pd(reg0, reg10);
            reg1 = _mm512_add_pd(reg1, reg11);
            reg2 = _mm_add_pd(reg2, reg12);

            reg0 = _mm512_exp_pd(reg0);
            reg1 = _mm512_exp_pd(reg1);
            reg2 = _mm_exp_pd(reg2);

            auto regw0 = _mm512_load_pd(&w[0]);
            auto regw1 = _mm512_load_pd(&w[8]);
            auto regw2 = _mm_load_pd(&w[16]);

            reg0 = _mm512_mul_pd(regw0, reg0);
            reg1 = _mm512_mul_pd(regw1, reg1);
            reg2 = _mm_mul_pd(regw2, reg2);

            reg0 = _mm512_add_pd(reg0, reg1);
            reg0 = _mm512_add_pd(reg0, _mm512_castpd128_pd512(reg2));

            sum = horizontalAdd(reg0);
#else
            for (std::size_t j = 0; j < 18; ++j) {
                t = x + deltax * y[j];
                auto mul = w[j] * std::exp(a1 * (std::log(t) - lnmu) + b1 * (std::log(one - t) - lnmuc));
                sum += mul;
            }
#endif
            np::float_ res = sum * deltax * std::exp(a1 * lnmu + b1 * lnmuc - lbeta(a, b));
            return res > zero ? one - res : -res;
        }

        // Computes the regularized incomplete beta function
        np::float_ betainc(np::float_ a, np::float_ b, np::float_ x) {
            const np::float_ zero = 0.0;
            const np::float_ one = 1.0;

            if (x < zero || x > one) {
                throw std::runtime_error("x must be within 0..1");
            }

            if (x == zero || x == one) {
                return x;
            }

            if (a < zero) {
                throw std::runtime_error("a must be >=0");
            }

            if (b < zero) {
                throw std::runtime_error("b must be >=0");
            }

            if (a == zero) {
                return one;
            }

            if (a == one) {
                return one - std::pow(one - x, b);
            }

            if (b == zero) {
                return zero;
            }

            if (b == 1) {
                return std::pow(x, a);
            }

            if (a == b && a == 0.5) {
                return 2 / M_PI * std::atan(std::sqrt(x / (one - x)));
            }

            if (x > 0.5) {
                return one - betainc(b, a, one - x);
            }

            if (a > 3000 && b > 3000) {
                return gaussLegendreQuadrature(a, b, x);
            }

            return fract(a, b, x);
        }

        using Array = np::Array<np::float_>;
        Array betainc(const Array &a, const Array &b, const Array &x) {
            if (a.shape() != b.shape() || b.shape() != x.shape()) {
                throw std::runtime_error("Shapes must be the same");
            }
            Array result{a.shape()};
            for (np::Size i = 0; i < a.size(); ++i) {
                result.set(i, betainc(a.get(i), b.get(i), x.get(i)));
            }
            return result;
        }


    }// namespace special
}// namespace scipy
