#!/bin/bash
#
# build.sh — Build, package, and archive the SciPy project.
#
# This script performs a three-step pipeline:
#   1. Build the project with CMake in Release mode.
#   2. Assemble a distribution package by copying source files,
#      headers, samples, scripts, unit tests, and the compiled
#      shared library into a versioned directory.
#   3. Compress the package directory into a .tgz archive.
#
# The package version and name are sourced from build.properties.
#
# Usage:
#   ./scripts/build.sh [additional CMake arguments...]
#
# Environment:
#   build.properties — must define version_major, version_minor,
#                      version_patch, and package_name.

# Resolve the project root directory (parent of scripts/).
ROOT_DIR="$(readlink -f $(dirname $BASH_SOURCE)/..)"

# Load version and package metadata.
source ${ROOT_DIR}/build.properties

# --- Paths ----------------------------------------------------------------
PACKAGE_ROOT=${ROOT_DIR}/package
PACKAGE_VERSION=${version_major}.${version_minor}.${version_patch}
PACKAGE_NAME=${package_name}
PACKAGE_FULLNAME=${PACKAGE_NAME}-${PACKAGE_VERSION}
PACKAGE_PATH=${PACKAGE_ROOT}/${PACKAGE_FULLNAME}
PACKAGE_TAR=${PACKAGE_PATH}.tgz

echo "PACKAGE_TAR: $PACKAGE_TAR"

# --- Functions ------------------------------------------------------------

# Build the project with CMake.
# Removes any previous build directory, configures with CMake in Release
# mode (setting the install prefix to the package directory), then compiles.
# Accepts extra arguments forwarded to the CMake configure step.
function build_package() {
    rm -rf ${ROOT_DIR}/build || return 1
    mkdir -p ${ROOT_DIR}/build || return 1
    cd ${ROOT_DIR}/build || return 1
    cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${PACKAGE_PATH} "$@"
    cmake --build . --config Release -j
}

# Copy source/header files from a given path into the package directory.
# Only files with recognised extensions (.hpp, .cpp, .md, .csv, .npy,
# .sh, .txt) at the top level of the source directory are copied.
# Arguments:
#   $1 — source directory (relative to ROOT_DIR)
#   $2 — destination directory (defaults to PACKAGE_PATH/$1)
function copy() {
    local path="$1"
    local dest="${2:-$PACKAGE_PATH/$path}"

    mkdir -p $dest

    find $path -maxdepth 1 -type f -regex ".*\.\(hpp\|cpp\|md\|csv\|npy\|sh\|txt\)$" -exec cp {} $dest \;
}

# Assemble the distribution package directory.
# Creates the versioned package directory, populates it by copying
# relevant folders (headers, samples, scripts, unit tests), and
# places the compiled shared library under lib/.
# Also creates a symlink <package-name> -> <package-name>-<version>
# for convenient access.
function create_package() {
    rm -rf ${PACKAGE_PATH}
    rm -f ${PACKAGE_TAR}
    mkdir -p ${PACKAGE_PATH}
    cd ${PACKAGE_ROOT} || return 1
    rm -f ${PACKAGE_NAME}
    ln -s ${PACKAGE_FULLNAME} ${PACKAGE_NAME}
    cd ${ROOT_DIR} || return 1

    FOLDERS=(
        .
        include
        include/scipy
        include/scipy/stats
        samples
        samples/stats
        scripts
        unit_tests
        unit_tests/include
        unit_tests/src
    )
    for folder in "${FOLDERS[@]}"; do
        copy $folder
    done
    mkdir -p $PACKAGE_PATH/lib
    copy build/src/libscipy.* $PACKAGE_PATH/lib

    return 0
}

# Compress the package directory into a .tgz archive.
function zip_package() {
    rm -f ${PACKAGE_TAR} || return 1
    tar zcf ${PACKAGE_TAR} -C "$(dirname ${PACKAGE_PATH})" "$(basename ${PACKAGE_PATH})"

    return 0
}

# --- Entry point ----------------------------------------------------------
function main() {
    build_package || return 1
    create_package || return 1
    zip_package
}

main
