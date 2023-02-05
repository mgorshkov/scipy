cmake_minimum_required(VERSION 3.13.0)

set(PROJECT stats)

project(${PROJECT})

set(CMAKE_CXX_STANDARD 17)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if (NOT DEFINED CMAKE_INSTALL_BINDIR)
    set(CMAKE_INSTALL_BINDIR "bin" CACHE PATH "user executables (bin)")
endif()

if (MSVC)
    add_compile_options(/W4 /WX)
	add_definitions(-DNOMINMAX) # allow using min/max in the project
else()
    add_compile_options(-Wall -Wextra -pedantic -Werror)
endif()

include(FetchContent)

FetchContent_Declare(
    np
    URL https://mgorshkov.jfrog.io/artifactory/default-generic-local/np/np-0.0.4.tgz
)

FetchContent_MakeAvailable(np)

include_directories(../../include ${np_SOURCE_DIR}/include)

add_executable(${PROJECT})

target_sources(${PROJECT} PUBLIC main.cpp)

target_link_libraries(
    ${PROJECT}
    ${PTHREAD})

install(
    FILES ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT}
    DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT ${PROJECT}
)
