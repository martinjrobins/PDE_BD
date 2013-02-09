# The name of our project is "PDE_BD". CMakeLists files in this project can 
# refer to the root source directory of the project as ${PDE_BD_SOURCE_DIR} and 
# to the root binary directory of the project as ${PDE_BD_BINARY_DIR}. 
project (pde_bd) 
cmake_minimum_required (VERSION 2.6) 
CMAKE_MINIMUM_REQUIRED(VERSION 2.6)
IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE "DEBUG")
  #SET(CMAKE_BUILD_TYPE "RELEASE")
  #SET(CMAKE_BUILD_TYPE "RELWITHDEBINFO")
  #SET(CMAKE_BUILD_TYPE "MINSIZEREL")
ENDIF()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${pde_bd_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${pde_bd_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${pde_bd_BINARY_DIR}/bin)

# Appends the cmake/modules path inside the MAKE_MODULE_PATH variable which stores the
# directories of additional CMake modules (ie. MacroOutOfSourceBuild.cmake):
set(CMAKE_MODULE_PATH ${pde_bd_SOURCE_DIR}/cmake/modules ${CMAKE_MODULE_PATH})

# The macro below forces the build directory to be different from source directory:
include(MacroOutOfSourceBuild)

# Recurse into the "Hello" and "Demo" subdirectories. This does not actually 
# cause another cmake executable to run. The same process will walk through 
# the project's entire directory structure. 
add_subdirectory (src)