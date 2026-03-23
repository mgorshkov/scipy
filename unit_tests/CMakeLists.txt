cmake_minimum_required(VERSION 3.13.0)

set(PROJECT_NAME scipy_unit_tests)

project(${PROJECT_NAME} VERSION 0.0.5 DESCRIPTION "SciPy Unit Tests")

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

add_executable(${PROJECT_NAME})

target_include_directories(${PROJECT_NAME}
    PRIVATE
        ../include
        include
)

target_include_directories(${PROJECT_NAME}
    SYSTEM PRIVATE
        ${googletest_SOURCE_DIR}/googletest/include
)

file(GLOB_RECURSE SOURCES ${PROJECT_SOURCE_DIR} "src/*.cpp")
target_sources(${PROJECT_NAME} PUBLIC ${SOURCES})

add_test(NAME ${PROJECT_NAME} COMMAND tests)

set (LIBS
    np
    pd
    gtest
    pthread)

# Apply target-dependent cmake options
include(${cmake_SOURCE_DIR}/Target.cmake)

install(
    FILES ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}
    DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT ${PROJECT_NAME}
)
