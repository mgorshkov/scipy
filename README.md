# About
Scientific methods on top of NP library.

# Requirements
Any C++17-compatible compiler:
* gcc 8 or higher
* clang 6 or higher
* Visual Studio 2017 or higher

# Repo
```
git clone https://github.com/mgorshkov/scipy.git
```

# Build unit tests and sample
```
mkdir build && cd build
cmake ..
cmake --build .
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
# How to build the sample

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
mkdir -p build-release && cd build-release
```
4. Configure cmake
```
cmake -DCMAKE_BUILD_TYPE=Release ..
```
5. Build
```
cmake --build .
```
6. Run the app
```
$./stats

```
