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

#include <gtest/gtest.h>

#include <scipy/special/betainc.hpp>

#include <ScipyTest.hpp>

using namespace scipy::special;

class SpecialTest : public ScipyTest {
protected:
};

TEST_F(SpecialTest, betaincTest) {
    {
        EXPECT_FLOAT_EQ(1.0, betainc(0.2, 3.5, 1.0));
    }
    {
        np::float_ a = 1.4, b = 3.1, x = 0.5;
        EXPECT_FLOAT_EQ(0.8148904036225296, betainc(a, b, x));
    }
    {
        np::float_ a = 0.5 * 99997;
        np::float_ b = 0.5 * 99997;
        np::float_ x = 0.49999;
        EXPECT_FLOAT_EQ(0.49747692843747587, betainc(a, b, x));
        x = 0.55;
        EXPECT_FLOAT_EQ(1.0, betainc(a, b, x));
        x = 0.56;
        EXPECT_FLOAT_EQ(1.0, betainc(a, b, x));
    }
}

TEST_F(SpecialTest, betaincArrayTest) {
    // Test array version with a few values
    np::Array<np::float_> a = {0.5, 1.4, 2.0, 10.0, 0.1};
    np::Array<np::float_> b = {0.5, 3.1, 3.0, 10.0, 0.1};
    np::Array<np::float_> x = {0.25, 0.5, 0.5, 0.5, 0.5};
    np::Array<np::float_> expected = {1.0 / 3.0,
                                      0.8148904036225296,
                                      0.6875,
                                      0.5,
                                      0.5};
    auto result = betainc(a, b, x);
    ASSERT_EQ(result.shape(), a.shape());
    for (np::Size i = 0; i < a.size(); ++i) {
        EXPECT_NEAR(result.get(i), expected.get(i), 1e-12);
    }
}

TEST_F(SpecialTest, betaincComprehensive) {
    // Edge cases
    EXPECT_FLOAT_EQ(1.0, betainc(0.0, 1.0, 0.5));
    EXPECT_FLOAT_EQ(0.0, betainc(1.0, 0.0, 0.5));
    EXPECT_TRUE(std::isnan(betainc(0.0, 0.0, 0.5)));
    EXPECT_FLOAT_EQ(0.0, betainc(0.5, 0.5, 0.0));
    EXPECT_FLOAT_EQ(1.0, betainc(0.5, 0.5, 1.0));
    EXPECT_NEAR(betainc(0.5, 0.5, 0.25), 1.0 / 3.0, 1e-15);
    EXPECT_FLOAT_EQ(0.0, betainc(2.0, 3.0, 0.0));
    EXPECT_FLOAT_EQ(1.0, betainc(2.0, 3.0, 1.0));
    EXPECT_FLOAT_EQ(0.6875, betainc(2.0, 3.0, 0.5));
    EXPECT_NEAR(betainc(10.0, 10.0, 0.5), 0.5, 1e-14);
    EXPECT_NEAR(betainc(0.1, 0.1, 0.5), 0.5, 1e-14);
    EXPECT_NEAR(betainc(100.0, 100.0, 0.49), 0.38877330806674637, 1e-12);
    EXPECT_NEAR(betainc(1000.0, 1000.0, 0.5), 0.5, 1e-12);
    EXPECT_NEAR(betainc(3001.0, 3001.0, 0.5), 0.5, 1e-11);
    EXPECT_NEAR(betainc(5000.0, 5000.0, 0.5), 0.5, 1e-11);
    EXPECT_NEAR(betainc(1e-10, 1e-10, 0.5), 0.5, 1e-14);
    // Large parameters - quadrature may have larger error
    EXPECT_NEAR(betainc(1e10, 1e10, 0.5), 0.5, 2e-5);
    // Random cases with moderate parameters
    EXPECT_NEAR(betainc(0.5, 2.0, 0.3), 0.7394254526319747, 1e-12);
    EXPECT_NEAR(betainc(2.0, 0.5, 0.7), 0.2605745473680253, 1e-12);
    EXPECT_NEAR(betainc(1.0, 1.0, 0.5), 0.5, 1e-14);
    // Symmetry property: I_x(a,b) = 1 - I_{1-x}(b,a)
    {
        np::float_ a = 2.5, b = 3.5, x = 0.3;
        np::float_ val1 = betainc(a, b, x);
        np::float_ val2 = 1.0 - betainc(b, a, 1.0 - x);
        EXPECT_NEAR(val1, val2, 1e-12);
    }
}
