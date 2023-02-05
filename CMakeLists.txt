cmake_minimum_required(VERSION 3.13.0)

set(SCIPY scipy)

project(${SCIPY} VERSION 0.0.1 DESCRIPTION "SciPy Methods on top of NP library")

set(CMAKE_CXX_STANDARD 17)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if (MSVC)
    add_compile_options(/W4 /WX)
    add_definitions(-DNOMINMAX) # allow using min/max in the ML
else()
    add_compile_options(-Wall -Wextra -pedantic -Werror)
endif()

include(FetchContent)

FetchContent_Declare(
    np
    URL https://mgorshkov.jfrog.io/artifactory/default-generic-local/np/np-0.0.4.tgz
)

FetchContent_MakeAvailable(np)

include_directories(include ${np_SOURCE_DIR}/include)

add_library(${SCIPY} INTERFACE)

# Make unit tests and doc if it's the main project
if (CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    set(MAKE_UNIT_TESTS ON)
    set(MAKE_DOC ON)
    set(MAKE_SAMPLE ON)
else()
    set(MAKE_UNIT_TESTS OFF)
    set(MAKE_DOC OFF)
    set(MAKE_SAMPLE OFF)
endif()

if (MAKE_UNIT_TESTS)
    message(STATUS "Enable testing in ${SCIPY}")
    add_subdirectory(unit_tests)
    include (CTest)
    enable_testing()
endif()

if (MAKE_DOC)
    message(STATUS "Enable doxygen in ${SCIPY}")
    add_subdirectory(doc)
endif()

if (MAKE_SAMPLE)
    message(STATUS "Enable samples build in ${SCIPY}")
    add_subdirectory(samples/stats)
endif()

install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/scipy
    DESTINATION include
    COMPONENT ${SCIPY_UNIT_TESTS}
)
