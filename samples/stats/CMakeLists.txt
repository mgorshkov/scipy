cmake_minimum_required(VERSION 3.13.0)

set(STATS stats)

project(${STATS})

# C++ 20
set(CMAKE_CXX_STANDARD 20)

include(FetchContent)

FetchContent_Declare(
    np
    GIT_REPOSITORY https://github.com/mgorshkov/np.git
    GIT_TAG main
)

FetchContent_MakeAvailable(np)

FetchContent_Declare(
    pd
    GIT_REPOSITORY https://github.com/mgorshkov/pd.git
    GIT_TAG main
)

FetchContent_MakeAvailable(pd)

include(FetchContent)
FetchContent_Declare(
    fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG 9.1.0
)
FetchContent_MakeAvailable(fmt)

include_directories(../../include ${np_SOURCE_DIR}/include ${pd_SOURCE_DIR}/include)

add_executable(${STATS})

target_sources(${STATS} PUBLIC main.cpp)

target_link_libraries(
    ${STATS}
    pd
    fmt::fmt
    ${PTHREAD})

install(
    TARGETS ${STATS}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT ${STATS}
)
