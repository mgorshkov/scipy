cmake_minimum_required(VERSION 3.13.0)

set(PROJECT_NAME betainc)

project(${PROJECT_NAME})

# Tune up various optimizations
# Enable CUDA support
option(ENABLE_CUDA "Enable CUDA optimizations" ON)
# Set up options for basic compile optimizations
option(ENABLE_AVX2 "Enable AVX2 optimizations" ON)
option(ENABLE_AVX512 "Enable AVX512 optimizations" OFF)
option(ENABLE_AMX "Enable AMX (Advanced Matrix Extensions) optimizations" OFF)
# Set up options for OpenMP support
option(ENABLE_OPENMP "Enable OpenMP" ON)

option(BOOST "Enable comparison with Boost library" ON)

include(FetchContent)

FetchContent_Declare(
    cmake
    GIT_REPOSITORY https://github.com/mgorshkov/cmake.git
    GIT_TAG main
)
FetchContent_MakeAvailable(cmake)

# Apply global cmake options
include(${cmake_SOURCE_DIR}/Config.cmake)

FetchContent_Declare(
    np
    GIT_REPOSITORY https://github.com/mgorshkov/np.git
    GIT_TAG main
    CMAKE_ARGS
    -DENABLE_CUDA=${ENABLE_CUDA}
    -DENABLE_OPENMP=${ENABLE_OPENMP}
    -DENABLE_SSE2=${ENABLE_SSE2}
    -DENABLE_SSE3=${ENABLE_SSE3}
    -DENABLE_AVX=${ENABLE_AVX}
    -DENABLE_AVX2=${ENABLE_AVX2}
    -DENABLE_AVX512=${ENABLE_AVX512}
    -DENABLE_AMX=${ENABLE_AMX}
    -DBUILD_SHARED_LIBS=${BUILD_SHARED_LIBS}
)

FetchContent_MakeAvailable(np)

FetchContent_Declare(
    scipy
    GIT_REPOSITORY https://github.com/mgorshkov/scipy.git
    GIT_TAG main
    CMAKE_ARGS
    -DENABLE_CUDA=${ENABLE_CUDA}
    -DENABLE_OPENMP=${ENABLE_OPENMP}
    -DENABLE_SSE2=${ENABLE_SSE2}
    -DENABLE_SSE3=${ENABLE_SSE3}
    -DENABLE_AVX=${ENABLE_AVX}
    -DENABLE_AVX2=${ENABLE_AVX2}
    -DENABLE_AVX512=${ENABLE_AVX512}
    -DENABLE_AMX=${ENABLE_AMX}
    -DBUILD_SHARED_LIBS=${BUILD_SHARED_LIBS}
    -DDEBUG_INFO=${DEBUG_INFO}
)

FetchContent_MakeAvailable(scipy)

add_executable(${PROJECT_NAME} main.cpp)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${np_SOURCE_DIR}/include
    ${scipy_SOURCE_DIR}/include)

if (${BOOST})
    add_compile_definitions(BOOST)

    FetchContent_Declare(
        boost
        GIT_REPOSITORY https://github.com/boostorg/math.git
        GIT_SHALLOW TRUE
    )

    FetchContent_MakeAvailable(boost)

    target_include_directories(${PROJECT_NAME} PRIVATE
        ${boost_SOURCE_DIR}/include)
endif()

# Comparison benchmark: C++ scipy (AVX2) vs Python scipy
set(COMPARISON_PROJECT_NAME ${PROJECT_NAME}_comparison)

add_executable(${COMPARISON_PROJECT_NAME} comparison_main.cpp)

target_include_directories(${COMPARISON_PROJECT_NAME} PRIVATE
    ${np_SOURCE_DIR}/include
    ${scipy_SOURCE_DIR}/include)

target_compile_definitions(${COMPARISON_PROJECT_NAME} PRIVATE
    PYTHON_BENCHMARK_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

# Apply target-dependent cmake options
include(${cmake_SOURCE_DIR}/Target.cmake)

install(
    FILES ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}
    DESTINATION "bin"
    COMPONENT ${PROJECT_NAME}
)

install(
    FILES ${CMAKE_CURRENT_BINARY_DIR}/${COMPARISON_PROJECT_NAME}
    DESTINATION "bin"
    COMPONENT ${PROJECT_NAME}
)
