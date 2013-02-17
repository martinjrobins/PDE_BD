file(GLOB pde_bd_SOURCES "*.cpp")
include_directories(${PDE_BD_INCLUDE_DIRECTORIES})


add_library(pdb SHARED ${pde_bd_SOURCES})
TARGET_LINK_LIBRARIES(pdb ${VTK_LIBRARIES} ${MPI_CXX_LIBRARIES} ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES})
