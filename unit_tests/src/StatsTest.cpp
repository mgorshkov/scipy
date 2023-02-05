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

#include <np/Comp.hpp>
#include <scipy/stats/mode.hpp>

#include <ScipyTest.hpp>

using namespace scipy::stats;

class StatsTest : public ScipyTest {
protected:
};

TEST_F(StatsTest, mode1DArrayTest) {
    {
        np::Array<np::int_> array{1, 2, 3, 3};
        auto result = mode<np::int_>(array);
        compare(result.first, np::Array<np::int_>{3});
        compare(result.second, np::Array<np::Size>{2});
    }
    {
        np::Array<np::int_> array{1, 2, 3};
        auto result = mode<np::int_>(array);
        compare(result.first, np::Array<np::int_>{1});
        compare(result.second, np::Array<np::Size>{1});
    }
    {
        np::Array<np::int_> array{3, 2, 1};
        auto result = mode<np::int_>(array);
        compare(result.first, np::Array<np::int_>{1});
        compare(result.second, np::Array<np::Size>{1});
    }
}

TEST_F(StatsTest, mode2DArrayTest) {
    np::int_ c_array[5][4] = {
            {6, 8, 3, 0},
            {3, 2, 1, 7},
            {8, 1, 8, 4},
            {5, 3, 0, 5},
            {4, 7, 5, 9}};
    Array<np::int_> array{c_array};
    np::int_ c_array_mode[1][4] = {{3, 1, 0, 0}};
    Array<np::int_> arrayMode{c_array_mode};
    np::Size c_array_count[1][4] = {{1, 1, 1, 1}};
    Array<np::Size> arrayCount{c_array_count};
    auto result = mode<np::int_>(array);
    compare(result.first, arrayMode);
    compare(result.second, arrayCount);
}
