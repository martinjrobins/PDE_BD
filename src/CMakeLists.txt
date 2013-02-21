file(GLOB pde_bd_SOURCES "*.cpp")
list(REMOVE_ITEM pde_bd_SOURCES ${pde_bd_SOURCE_DIR}/src/Molecules.cpp)
message(STATUS ${pde_bd_SOURCES})
include_directories(${PDE_BD_INCLUDE_DIRECTORIES})


add_library(pdb SHARED ${pde_bd_SOURCES})
TARGET_LINK_LIBRARIES(pdb "/scratch/robinsonm/pkg/lib/libtrng4.so" ${VTK_LIBRARIES} ${MPI_CXX_LIBRARIES} ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES})
