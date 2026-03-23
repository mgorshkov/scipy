file(GLOB SOURCE_FILES "*.cpp")

# Create the library
add_library(${PROJECT_NAME} ${SOURCE_FILES})
