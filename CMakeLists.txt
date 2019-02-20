cmake_minimum_required(VERSION 2.8)
project(likelihood)

cmake_policy(SET CMP0015 NEW)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY lib)

set(CMAKE_C_STANDARD 99)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

FIND_PACKAGE(OpenMP REQUIRED)
if (OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif ()

file(GLOB SOURCES src/*.h src/*.c)
add_library(likelihood SHARED ${SOURCES})

get_filename_component(LIB_PATH lib REALPATH)
find_library(CHRONOS_LIB chronos ${LIB_PATH})
target_link_libraries(likelihood ${CHRONOS_LIB} gsl)