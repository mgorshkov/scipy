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
