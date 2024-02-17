/*
Scientific methods on top of NP library

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

#include <np/Array.hpp>
#include <np/linalg/LstSq.hpp>

#pragma once

namespace scipy {
    namespace linalg {
        // Compute least-squares solution to equation Ax = b my MRRR algorithm.
        template<np::Arithmetic DType1, typename Derived1, typename Storage1, np::Arithmetic DType2, typename Derived2, typename Storage2>
        inline auto lstsq(const np::ndarray::internal::NDArrayBase<DType1, Derived1, Storage1> &a, const np::ndarray::internal::NDArrayBase<DType2, Derived2, Storage2> &b) {
            if (a.ndim() != 2) {
                throw std::runtime_error("Array a should be 2D");
            }
            if (a.shape()[0] != b.shape()[0]) {
                throw std::runtime_error("a and b should have the same number of rows");
            }
            return np::linalg::lstsq_mrrr(a, b);
        }
    }// namespace linalg
}// namespace scipy
