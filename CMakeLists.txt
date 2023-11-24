cmake_minimum_required(VERSION 3.12)
project(casadi-tutorial)

# for color output only
if(NOT WIN32)
  string(ASCII 27 Esc)
  set(ColourReset "${Esc}[m")
  set(ColourBold  "${Esc}[1m")
  set(Red         "${Esc}[31m")
  set(Green       "${Esc}[32m")
  set(Yellow      "${Esc}[33m")
  set(Blue        "${Esc}[34m")
  set(Magenta     "${Esc}[35m")
  set(Cyan        "${Esc}[36m")
  set(White       "${Esc}[37m")
  set(BoldRed     "${Esc}[1;31m")
  set(BoldGreen   "${Esc}[1;32m")
  set(BoldYellow  "${Esc}[1;33m")
  set(BoldBlue    "${Esc}[1;34m")
  set(BoldMagenta "${Esc}[1;35m")
  set(BoldCyan    "${Esc}[1;36m")
  set(BoldWhite   "${Esc}[1;37m")
endif()

SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath -Wl,${CMAKE_SOURCE_DIR}/build")

# only packages we'll use
find_package(casadi REQUIRED)
find_package(Threads REQUIRED)

#set cmake type
set(CMAKE_BUILD_TYPE "Release")

set(CMAKE_CXX_FLAGS "-Wall -Wextra -pthread")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

message(STATUS "Compiler Path: ${CMAKE_CXX_COMPILER}")
message(STATUS "Compiler ID: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "Compiler Version: ${CMAKE_CXX_COMPILER_VERSION}")

#ipopt dir
set(LIBRARY_DIRS /usr/local/lib)

# fidn casadi
set(CASADI_INCLUDE_DIR /usr/local/include)
find_library(CASADI_LIBRARY NAME casadi HINTS ${CASADI_INCLUDE_DIR}/../lib $ENV{CASADI_PREFIX}/lib)
if (CASADI_LIBRARY)
    set(CASADI_LIBRARIES ${CASADI_LIBRARIES} ${CASADI_LIBRARY})
endif()

find_package(Threads REQUIRED)

include_directories(
    ${CMAKE_SOURCE_DIR}/include
    SYSTEM ${IPOPT_INCLUDE_DIRS}
    SYSTEM ${CASADI_INCLUDE_DIR}
    ${CODE_GEN_INCLUDE_DIR}
)

#library directories
link_directories(
    ${LIBRARY_DIRS}
)

#find all header files
file(GLOB HEADER_FILES_HPP ${CMAKE_SOURCE_DIR}/include/*.hpp)
file(GLOB HEADER_FILES_H ${CMAKE_SOURCE_DIR}/include/*.h)

# NOTE: actual files we'll use for the project
###############################################


# casadi basic operations
add_executable(example_basic_operations example/example_basic_operations.cpp)
target_link_libraries(example_basic_operations ${CASADI_LIBRARIES})

# #run test to see if ipopt and casadi work
add_executable(example_racecar example/example_racecar.cpp)
target_link_libraries(example_racecar ${CASADI_LIBRARIES} ipopt)

# solve a NLP given a compiled shared library
add_executable(example_use_nlp_external example/example_use_nlp_external.cpp)
target_link_libraries(example_use_nlp_external ${CASADI_LIBRARIES})

###############################################