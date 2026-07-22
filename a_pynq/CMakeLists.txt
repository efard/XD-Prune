cmake_minimum_required(VERSION 3.16)

project(yolo26_ncnn LANGUAGES CXX)

# std::filesystem is used to validate the model deployment files.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ncnn_DIR must point to the directory containing ncnnConfig.cmake.
# Passing it on the CMake command line keeps machine-specific paths out of
# this project file.
find_package(ncnn CONFIG REQUIRED)

add_executable(yolo26_ncnn
    yolo26_ncnn.cpp
)

# The imported NCNN target supplies both include directories and libraries.
target_link_libraries(yolo26_ncnn
    PRIVATE
        ncnn
)

if(MSVC)
    # Enable useful warnings without changing inference behavior.
    target_compile_options(yolo26_ncnn PRIVATE /W4 /permissive-)
else()
    target_compile_options(yolo26_ncnn PRIVATE -Wall -Wextra -Wpedantic)
endif()
