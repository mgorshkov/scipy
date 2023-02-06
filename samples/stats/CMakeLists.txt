cmake_minimum_required(VERSION 3.13.0)

set(STATS stats)

project(${STATS})

include(FetchContent)

include_directories(../../include)

add_executable(${STATS})

target_sources(${STATS} PUBLIC main.cpp)

target_link_libraries(
    ${STATS}
    pd
    ${PTHREAD})

install(
    FILES ${CMAKE_CURRENT_BINARY_DIR}/${STATS}
    DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT ${STATS}
)
