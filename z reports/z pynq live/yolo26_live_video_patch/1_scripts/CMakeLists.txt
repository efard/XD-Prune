# PYNQ 2.5 uses CMake 3.10.2, so keep this project compatible with it.
cmake_minimum_required(VERSION 3.10)

project(yolo26_ncnn_runner LANGUAGES CXX)

# GCC 7.3 on the PYNQ-Z2 supports C++14 reliably.
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(
    NCNN_ROOT
    "/home/xilinx/yolo26_ps/third_party/ncnn/install"
    CACHE PATH
    "Installed NCNN directory"
)

set(NCNN_INCLUDE_DIR "${NCNN_ROOT}/include/ncnn")
set(NCNN_LIBRARY "${NCNN_ROOT}/lib/libncnn.a")

if(NOT EXISTS "${NCNN_INCLUDE_DIR}/net.h")
    message(FATAL_ERROR "NCNN header not found: ${NCNN_INCLUDE_DIR}/net.h")
endif()
if(NOT EXISTS "${NCNN_LIBRARY}")
    message(FATAL_ERROR "NCNN library not found: ${NCNN_LIBRARY}")
endif()

find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)

# Keep the validated image benchmark/validation runner available.
add_executable(yolo26_ncnn_runner yolo26_ncnn_runner.cpp)

# New TCP server used by the PC live-video client.
add_executable(yolo26_ncnn_server yolo26_ncnn_server.cpp)

foreach(TARGET_NAME yolo26_ncnn_runner yolo26_ncnn_server)
    target_include_directories(
        ${TARGET_NAME}
        PRIVATE
        "${NCNN_INCLUDE_DIR}"
        ${OpenCV_INCLUDE_DIRS}
    )

    target_compile_options(
        ${TARGET_NAME}
        PRIVATE
        -O3
        ${OpenMP_CXX_FLAGS}
    )

    target_link_libraries(
        ${TARGET_NAME}
        PRIVATE
        "${NCNN_LIBRARY}"
        ${OpenCV_LIBS}
        ${OpenMP_CXX_FLAGS}
        Threads::Threads
        dl
        m
        atomic
    )
endforeach()
