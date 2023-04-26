cmake_minimum_required(VERSION 3.11)
project(cudafm)
enable_language(CUDA)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

# We require the GNU compiler for this assignment.
if (NOT "${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
    set(Prg-Intel "PrgEnv-intel")
    set(Prg-Clang "PrgEnv-cray")
    message(WARNING
            "Must use GNU Compiler for submission. Make sure you ran:\n"
            "module swap ${Prg-${CMAKE_C_COMPILER_ID}} PrgEnv-gnu")
endif ()


add_executable(serial test.cpp fm.h fm.cpp fm_learn.h fm_learn.cpp Data.h Data.cpp util/matrix.h util/fmatrix.h util/random.h util/random.cpp)
target_compile_features(serial PRIVATE cxx_std_14)


# Copy the job scripts
configure_file(job-serial job-serial COPYONLY)
