include_directories(${PDE_BD_INCLUDE_DIRECTORIES})
LINK_DIRECTORIES(${pde_bd_BINARY_DIR}/lib)

add_executable (test_init test_init.cpp) 
target_link_libraries (test_init pdb) 

add_executable (test_pde_constructor test_pde_constructor.cpp) 
target_link_libraries (test_pde_constructor pdb)

enable_testing()
add_test(init ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test_init)
add_test(pde_constructor ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test_pde_constructor)
add_test(pde_constructor_2_cpu ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test_pde_constructor)