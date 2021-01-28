cmake_minimum_required(VERSION 3.5)
project(lanedetection)

find_package(Python REQUIRED)
find_package(OpenCV REQUIRED)

if (MSVC)
  message(STATUS "########## Eigen3 directory: ${EIGEN3_DIR}. ##########")
  set( CMAKE_MODULE_PATH "${EIGEN3_DIR}" )

  message(STATUS "########## Pytorch directory: ${TORCH_DIR}. ##########")
  set( CMAKE_MODULE_PATH "${TORCH_DIR}" )

  set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W0")
endif (MSVC)

find_package(Torch REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIR})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(SOURCE_FILES main.cpp laneprocessing.cpp
    findnonzero.cpp binaryframe.cpp
    erosiondilation.cpp yolodetection.cpp
    processingframe.cpp grayframe.cpp
    hsvframe.cpp wrapframe.cpp 
    polyfitpoints.cpp postprocessing.cpp 
    preprocessing.cpp resizeframe.cpp 
    sanitycheck.cpp sobelframe.cpp 
    videoprocessing.cpp laneyolo.hpp)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

add_executable(lanedetection "${SOURCE_FILES}")
target_link_libraries(lanedetection "${OpenCV_LIBS}" "${TORCH_LIBRARIES}" Eigen3::Eigen)
set_property(TARGET lanedetection PROPERTY CXX_STANDARD 14)

if (MSVC)
  message(STATUS "########## Windows env copying torch dll. ##########")
  file(GLOB TORCH_DLLS "${TORCH_DIR}/lib/*.dll")
  add_custom_command(TARGET lanedetection
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:lanedetection>)
endif (MSVC)

