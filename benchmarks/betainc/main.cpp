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

#include <iostream>

#include <scipy/special/betainc.hpp>
#include <functional>
#include <time.h>

#ifdef BOOST
#include <boost/math/special_functions/beta.hpp>
#endif
typedef double (*IncompleteBetafunc)(double, double, double);

void measureIncompleteBetaFunction(IncompleteBetafunc func) {
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

    std::cout << "Result = " << res << std::endl;
    std::cout << "Time = " << diff << " ns" << std::endl;
    std::cout << "Loops = " << count << std::endl;
}


int main(int, char **) {
#ifdef BOOST
    std::cout << "Boost incomplete beta function" << std::endl;
    measureIncompleteBetaFunction(boost::math::ibeta);
#endif
    std::cout << "Scipy incomplete beta function" << std::endl;
    measureIncompleteBetaFunction(scipy::special::betainc);

    return 0;
}
