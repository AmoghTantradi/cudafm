cmake_minimum_required(VERSION 3.11)
project(cudafm)
enable_language(CUDA)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O0")

# We require the GNU compiler for this assignment.
if (NOT "${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
    set(Prg-Intel "PrgEnv-intel")
    set(Prg-Clang "PrgEnv-cray")
    message(WARNING
            "Must use GNU Compiler for submission. Make sure you ran:\n"
            "module swap ${Prg-${CMAKE_C_COMPILER_ID}} PrgEnv-gnu")
endif ()

find_package(CUDAToolkit REQUIRED)




set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}$; -lcusparse)


add_executable(gpu main.cu fm.h fm.cu data.h data.cpp util/fmatrix.h util/random.h util/random.cpp)

target_link_libraries(gpu CUDA::cusparse)


target_compile_features(gpu PRIVATE cxx_std_14)



set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_80")

set_target_properties(gpu PROPERTIES CUDA_ARCHITECTURES "80")




# Copy the job scripts
configure_file(job-gpu job-gpu COPYONLY)
