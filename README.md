[![Build status](https://ci.appveyor.com/api/projects/status/bdews6m7dh2botlx/branch/main?svg=true)](https://ci.appveyor.com/project/mgorshkov/scipy/branch/main)

# About
Scientific methods on top of NP library.

# Requirements
Any C++20-compatible compiler:
* gcc 10 or higher
* clang 6 or higher
* Visual Studio 2019 or higher

# Repo
```
git clone https://github.com/mgorshkov/scipy.git
```

# Build unit tests and sample
## Linux/MacOS
```
mkdir build && cd build
cmake ..
cmake --build .
```
## Windows
```
mkdir build && cd build
cmake ..
cmake --build . --config Release
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
## Linux/MacOS
```
cmake --build .
```
## Windows
```
cmake --build . --config Release
```
6. Run the app
```
$./stats

```

# Links
* C++ numpy-like template-based array implementation: https://github.com/mgorshkov/np
* Methods from pandas library on top of NP library: https://github.com/mgorshkov/pd
* ML Methods from scikit-learn library: https://github.com/mgorshkov/sklearn
