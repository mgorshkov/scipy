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

#include <np/Array.hpp>

namespace scipy {
    namespace special {
        np::float_ lbeta(np::float_ a, np::float_ b) {
            return std::lgamma(a) + std::lgamma(b) - std::lgamma(a + b);
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

            if (x < (a + 1.0) / (a + b + 2.0)) {
                return res * fr(a, b, x) / a;
            } else {
                return 1.0 - res * fr(b, a, 1.0 - x) / b;
            }
        }

        np::float_ gaussLegendreQuadrature(np::float_ a, np::float_ b, np::float_ x) {
            np::float_ y[] = {0.0021695375159141994,
                              0.011413521097787704, 0.027972308950302116, 0.051727015600492421,
                              0.082502225484340941, 0.12007019910960293, 0.16415283300752470,
                              0.21442376986779355, 0.27051082840644336, 0.33199876341447887,
                              0.39843234186401943, 0.46931971407375483, 0.54413605556657973,
                              0.62232745288031077, 0.70331500465597174, 0.78649910768313447,
                              0.87126389619061517, 0.95698180152629142};
            np::float_ w[] = {0.0055657196642445571,
                              0.012915947284065419, 0.020181515297735382, 0.027298621498568734,
                              0.034213810770299537, 0.040875750923643261, 0.047235083490265582,
                              0.053244713977759692, 0.058860144245324798, 0.064039797355015485,
                              0.068745323835736408, 0.072941885005653087, 0.076598410645870640,
                              0.079687828912071670, 0.082187266704339706, 0.084078218979661945,
                              0.085346685739338721, 0.085983275670394821};

            np::float_ mu = a / (a + b);
            np::float_ lnmu = std::log(mu);
            np::float_ lnmuc = std::log(1.0 - mu);
            np::float_ t = std::sqrt(a * b / ((a + b) * (a + b) * (a + b + 1.0)));
            np::float_ xu;
            if (x > a / (a + b)) {
                xu = std::min(1.0, std::max(mu + 10.0 * t, x + 5.0 * t));
            } else {
                xu = std::max(0.0, std::min(mu - 10.0 * t, x - 5.0 * t));
            }

            np::float_ sum = 0;
            np::float_ a1 = a - 1.0;
            np::float_ b1 = b - 1.0;
            for (std::size_t j = 0; j < 18; ++j) {
                t = x + (xu - x) * y[j];
                sum += w[j] * std::exp(a1 * (std::log(t) - lnmu) + b1 * (std::log(1 - t) - lnmuc));
            }

            np::float_ res = sum * (xu - x) * std::exp(a1 * lnmu + b1 * lnmuc - lbeta(a, b));
            return res > 0.0 ? 1.0 - res : -res;
        }

        // Computes the regularized incomplete beta function
        np::float_ betainc(np::float_ a, np::float_ b, np::float_ x) {
            if (x < 0.0 or x > 1.0) {
                throw std::runtime_error("x must be within 0..1");
            }

            if (x == 0.0 || x == 1.0) {
                return x;
            }

            if (a < 0) {
                throw std::runtime_error("a must be >=0");
            }

            if (b < 0) {
                throw std::runtime_error("b must be >=0");
            }

            if (a == 0) {
                return 1.0;
            }

            if (a == 1) {
                return 1 - std::pow(1 - x, b);
            }

            if (b == 0) {
                return 0.0;
            }

            if (b == 1) {
                return std::pow(x, a);
            }

            if (a == b && a == 0.5) {
                return 2 / M_PI * std::atan(std::sqrt(x / (1 - x)));
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