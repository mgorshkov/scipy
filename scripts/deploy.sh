#!/bin/bash
#
# deploy.sh — Upload the pre-built .tgz package to a remote Artifactory
#             repository.
#
# This script reads the package name and version from build.properties,
# constructs the expected archive path under package/, and uploads it
# to the configured Artifactory server using HTTP PUT (curl).
#
# Prerequisites:
#   - build.sh must have been run first so that the .tgz archive exists.
#   - The environment variables USERNAME and PASSWORD must be set to
#     valid Artifactory credentials.
#
# Usage:
#   USERNAME=myuser PASSWORD=mypass ./scripts/deploy.sh
#
# Environment:
#   build.properties — must define version_major, version_minor,
#                      version_patch, and package_name.
#   USERNAME         — Artifactory username (must be exported).
#   PASSWORD         — Artifactory password or API token (must be exported).

# Resolve the project root directory (parent of scripts/).
ROOT_DIR="$(readlink -f $(dirname $BASH_SOURCE)/..)"

# Load version and package metadata.
source ${ROOT_DIR}/build.properties

# --- Paths ----------------------------------------------------------------
PACKAGE_ROOT=${ROOT_DIR}/package
PACKAGE_VERSION=${version_major}.${version_minor}.${version_patch}
PACKAGE_NAME=${package_name}
PACKAGE_FULLNAME=${PACKAGE_NAME}-${PACKAGE_VERSION}.tgz
PACKAGE_PATH=${PACKAGE_ROOT}/${PACKAGE_FULLNAME}

# Artifactory upload URL for the generic repository.
URL=https://mgorshkov.jfrog.io/artifactory/default-generic-local/scipy/$PACKAGE_FULLNAME

# Upload the archive using HTTP PUT.
# Credentials are supplied via USERNAME and PASSWORD environment variables.
curl -T $PACKAGE_PATH -u$USERNAME:$PASSWORD "$URL"
