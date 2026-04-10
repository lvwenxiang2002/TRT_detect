cmake_minimum_required(VERSION 3.16)
project(TRTInferenceGUI VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

# ── Qt5 ──────────────────────────────────────────────────────────────────────
find_package(Qt5 REQUIRED COMPONENTS Core Gui Widgets Concurrent)

# ── CUDA 11.8 ─────────────────────────────────────────────────────────────────
find_package(CUDAToolkit REQUIRED)

# ── TensorRT 8.5 ──────────────────────────────────────────────────────────────
set(TRT_ROOT "C:/TensorRT-8.5.1.7" CACHE PATH "TensorRT installation root")

find_path(TRT_INCLUDE_DIR
    NAMES NvInfer.h
    PATHS ${TRT_ROOT}/include
    REQUIRED
)
find_library(TRT_LIB        nvinfer        PATHS ${TRT_ROOT}/lib REQUIRED)
find_library(TRT_PLUGIN_LIB nvinfer_plugin PATHS ${TRT_ROOT}/lib)

# ── OpenCV (可选) ──────────────────────────────────────────────────────────────
find_package(OpenCV QUIET)
if(OpenCV_FOUND)
    message(STATUS "OpenCV ${OpenCV_VERSION} found")
    add_compile_definitions(WITH_OPENCV)
else()
    message(WARNING "OpenCV not found – using stb_image fallback")
endif()

# ── Sources ───────────────────────────────────────────────────────────────────
set(SOURCES
    src/main.cpp
    src/mainwindow.cpp
    src/enginetaskwidget.cpp
    src/trtengine.cpp
    src/inferenceworker.cpp
)

add_compile_options("$<$<C_COMPILER_ID:MSVC>:/utf-8>")
add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")
add_executable(TRTInferenceGUI ${SOURCES})

target_include_directories(TRTInferenceGUI PRIVATE
    src
    ${TRT_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(TRTInferenceGUI PRIVATE
    Qt5::Core
    Qt5::Gui
    Qt5::Widgets
    Qt5::Concurrent
    CUDA::cudart
    ${TRT_LIB}
    $<$<BOOL:${TRT_PLUGIN_LIB}>:${TRT_PLUGIN_LIB}>
)

if(OpenCV_FOUND)
    target_include_directories(TRTInferenceGUI PRIVATE ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(TRTInferenceGUI PRIVATE ${OpenCV_LIBS})
endif()

if(WIN32)
    add_custom_command(TARGET TRTInferenceGUI POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${TRT_ROOT}/lib/nvinfer.dll"
            $<TARGET_FILE_DIR:TRTInferenceGUI>
    )
endif()