cmake_minimum_required(VERSION 3.13.0)

set(SCIPY_UNIT_TESTS scipy_unit_tests)

project(${SCIPY_UNIT_TESTS} VERSION 0.0.4 DESCRIPTION "SciPy Unit Tests")

include(FetchContent)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG main
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
set(BUILD_GTEST ON CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(googletest)

include_directories(../include include ${googletest_SOURCE_DIR}/googletest/include)

add_executable(${SCIPY_UNIT_TESTS})

file(GLOB_RECURSE SOURCES ${PROJECT_SOURCE_DIR} "src/*.cpp")
target_sources(${SCIPY_UNIT_TESTS} PUBLIC ${SOURCES})

add_test(NAME ${SCIPY_UNIT_TESTS} COMMAND tests)

target_link_libraries(
    ${SCIPY_UNIT_TESTS}
    pd
    gtest
    ${PTHREAD})

install(
    FILES ${CMAKE_CURRENT_BINARY_DIR}/${SCIPY_UNIT_TESTS}
    DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT ${SCIPY_UNIT_TESTS}
)