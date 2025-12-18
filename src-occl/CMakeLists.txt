cmake_minimum_required(VERSION 3.20)

project(CoMD_SYCL_oneCCL LANGUAGES C CXX)

# -----------------------------
# Options: mirror Makefile
# -----------------------------
option(DOUBLE_PRECISION "Double precision (ON/OFF)" ON)
option(DO_MPI           "MPI support (ON/OFF)"          ON)
option(USE_ONECCL       "oneCCL support (ON/OFF)"       ON)

set(MAXATOMS "256" CACHE STRING "Maximum number of atoms in a link cell")

# Mirrors Makefile knobs for MPI wrapper includes/libs (default empty)
set(MPI_INCLUDE "" CACHE STRING "Extra MPI include flags (e.g. -IC:/path/include)")
set(MPI_LIB     "" CACHE STRING "Extra MPI link flags (e.g. -LC:/path/lib -lmpi)")

# oneAPI root (Makefile default is Linux-only; here we set a platform default)
if(WIN32)
  set(_ONEAPI_DEFAULT "C:/Program Files (x86)/Intel/oneAPI")
else()
  set(_ONEAPI_DEFAULT "/opt/intel/oneapi")
endif()
set(ONEAPI_ROOT "${_ONEAPI_DEFAULT}" CACHE PATH "Intel oneAPI root path")

# oneCCL root mirrors Makefile: $(ONEAPI_ROOT)/ccl/latest
set(CCL_ROOT "${ONEAPI_ROOT}/ccl/latest" CACHE PATH "oneCCL root path")

# SYCL flags (mirror Makefile)
set(SYCL_FLAGS "-fsycl;-fsycl-targets=spir64" CACHE STRING "SYCL flags list (semicolon-separated)")

# OPTFLAGS mirror: -g -O3
set(OPTFLAGS "-g;-O3" CACHE STRING "Optimization/debug flags list (semicolon-separated)")

# -----------------------------
# Sources: mirror Makefile
# -----------------------------
set(C_SOURCES
  cmdLineParser.c
  decomposition.c
  hashTable.c
  initAtoms.c
  ljForce.c
  mycommand.c
  mytype.c
  neighborList.c
  performanceTimers.c
  random.c
  yamlOutput.c
)

set(CXX_SOURCES
  CoMD.cpp
  parallel.cpp
  gpu_utility.cpp
  gpu_kernels.cpp
  haloExchange.cpp
  linkCells.cpp
  eam.cpp
  timestep.cpp
  gpu_neighborList.cpp
)

# -----------------------------
# Variant name: mirror Makefile logic exactly
# -----------------------------
set(CoMD_VARIANT "CoMD-sycl")
if(DO_MPI)
  set(CoMD_VARIANT "CoMD-sycl-mpi")
endif()
if(USE_ONECCL)
  # In Makefile this unconditionally becomes mpi-ccl name when USE_ONECCL=ON
  set(CoMD_VARIANT "CoMD-sycl-mpi-ccl")
endif()

# Output directory mirrors Makefile BIN_DIR=../bin
set(BIN_DIR "${CMAKE_CURRENT_LIST_DIR}/../bin")

# -----------------------------
# CoMD_info.h generation (mirror generate_info_header rule)
# -----------------------------
set(GEN_INFO "${CMAKE_CURRENT_LIST_DIR}/generate_info_header")

# We will generate in build dir (recommended); code includes "CoMD_info.h"
set(COMD_INFO_H "${CMAKE_CURRENT_BINARY_DIR}/CoMD_info.h")

# Build-time strings passed to generator (close to Makefile's "$(CXXFLAGS)" "$(LDFLAGS)")
# We intentionally compose a Makefile-like view rather than CMake's internal flag model.
set(_CXXFLAGS_LIST
  "-std=c++17"
  "-DMAXATOMS=${MAXATOMS}"
)
set(_CFLAGS_LIST
  "-std=c99"
  "-Wno-unused-result"
  "-DMAXATOMS=${MAXATOMS}"
)

# Precision (Makefile adds COMD_DOUBLE/SINGLE to both CFLAGS and CXXFLAGS)
if(DOUBLE_PRECISION)
  list(APPEND _CFLAGS_LIST  "-DCOMD_DOUBLE")
  list(APPEND _CXXFLAGS_LIST "-DCOMD_DOUBLE")
else()
  list(APPEND _CFLAGS_LIST  "-DCOMD_SINGLE")
  list(APPEND _CXXFLAGS_LIST "-DCOMD_SINGLE")
endif()

# MPI (Makefile: adds -DDO_MPI + INCLUDES += MPI_INCLUDE + LDFLAGS += MPI_LIB)
if(DO_MPI)
  list(APPEND _CFLAGS_LIST  "-DDO_MPI")
  list(APPEND _CXXFLAGS_LIST "-DDO_MPI")
  if(NOT MPI_INCLUDE STREQUAL "")
    list(APPEND _CFLAGS_LIST  "${MPI_INCLUDE}")
    list(APPEND _CXXFLAGS_LIST "${MPI_INCLUDE}")
  endif()
endif()

# oneCCL (Makefile: adds -DUSE_ONECCL + -I... + -L... -lccl)
if(USE_ONECCL)
  list(APPEND _CXXFLAGS_LIST "-DUSE_ONECCL")
  list(APPEND _CXXFLAGS_LIST "-I${CCL_ROOT}/include")
endif()

# SYCL and OPTFLAGS (Makefile appends SYCL_FLAGS and OPTFLAGS)
list(APPEND _CXXFLAGS_LIST ${SYCL_FLAGS})
list(APPEND _CXXFLAGS_LIST ${OPTFLAGS})
list(APPEND _CFLAGS_LIST  ${OPTFLAGS})

# LDFLAGS string (Makefile: LDFLAGS += CCL_LIB (if enabled) + MPI_LIB (if MPI) + -lm)
set(_LDFLAGS_LIST)
if(DO_MPI AND NOT MPI_LIB STREQUAL "")
  list(APPEND _LDFLAGS_LIST "${MPI_LIB}")
endif()
if(USE_ONECCL)
  # Try common layouts: lib or lib/intel64
  if(EXISTS "${CCL_ROOT}/lib")
    list(APPEND _LDFLAGS_LIST "-L${CCL_ROOT}/lib")
  elseif(EXISTS "${CCL_ROOT}/lib/intel64")
    list(APPEND _LDFLAGS_LIST "-L${CCL_ROOT}/lib/intel64")
  endif()
  list(APPEND _LDFLAGS_LIST "-lccl")
endif()

# Math lib like Makefile (-lm) only meaningful on UNIX
if(UNIX)
  list(APPEND _LDFLAGS_LIST "-lm")
endif()

# Join lists into single strings for the generator
string(JOIN " " CXXFLAGS_STR ${_CXXFLAGS_LIST})
string(JOIN " " LDFLAGS_STR ${_LDFLAGS_LIST})

# On Windows, generate_info_header might be a bash script: allow running via bash if present.
find_program(BASH_EXECUTABLE bash)

if(WIN32 AND BASH_EXECUTABLE)
  set(_GEN_CMD ${BASH_EXECUTABLE} "-lc" "${GEN_INFO} ${CoMD_VARIANT} \"${CMAKE_CXX_COMPILER}\" \"${CXXFLAGS_STR}\" \"${LDFLAGS_STR}\" > \"${COMD_INFO_H}\"")
else()
  # Works on Linux/macOS; also on Windows if generate_info_header is a native .exe
  set(_GEN_CMD ${GEN_INFO} ${CoMD_VARIANT} "${CMAKE_CXX_COMPILER}" "${CXXFLAGS_STR}" "${LDFLAGS_STR}")
endif()

# Custom command: produce CoMD_info.h
add_custom_command(
  OUTPUT  "${COMD_INFO_H}"
  COMMAND ${CMAKE_COMMAND} -E echo_append ""  # no-op to keep COMMAND list valid across platforms
  COMMAND ${CMAKE_COMMAND} -E rm -f "${COMD_INFO_H}"
  # If we are using bash -lc redirection, the generator writes to COMD_INFO_H itself
  COMMAND ${_GEN_CMD}
  DEPENDS "${GEN_INFO}" "${CMAKE_CURRENT_LIST_FILE}"
  VERBATIM
)

add_custom_target(comd_info_header DEPENDS "${COMD_INFO_H}")

# -----------------------------
# Executable
# -----------------------------
add_executable(CoMD
  "${COMD_INFO_H}"
  ${C_SOURCES}
  ${CXX_SOURCES}
)

add_dependencies(CoMD comd_info_header)

# Include dirs: source + build (for CoMD_info.h)
target_include_directories(CoMD PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}"
  "${CMAKE_CURRENT_BINARY_DIR}"
)

# Definitions: MAXATOMS is in Makefile via -DMAXATOMS=...
target_compile_definitions(CoMD PRIVATE "MAXATOMS=${MAXATOMS}")

# Precision definitions (already in the composed strings, but also apply to build for correctness)
if(DOUBLE_PRECISION)
  target_compile_definitions(CoMD PRIVATE COMD_DOUBLE)
else()
  target_compile_definitions(CoMD PRIVATE COMD_SINGLE)
endif()

# MPI definition (mirrors Makefile)
if(DO_MPI)
  target_compile_definitions(CoMD PRIVATE DO_MPI)
endif()

# oneCCL definition (mirrors Makefile)
if(USE_ONECCL)
  target_compile_definitions(CoMD PRIVATE USE_ONECCL)
  target_include_directories(CoMD PRIVATE "${CCL_ROOT}/include")
  if(EXISTS "${CCL_ROOT}/lib")
    target_link_directories(CoMD PRIVATE "${CCL_ROOT}/lib")
  elseif(EXISTS "${CCL_ROOT}/lib/intel64")
    target_link_directories(CoMD PRIVATE "${CCL_ROOT}/lib/intel64")
  endif()
  # Link name: on Windows expects ccl.lib; on Linux expects -lccl
  target_link_libraries(CoMD PRIVATE ccl)
endif()

# Compile options: mirror OPTFLAGS and SYCL flags
target_compile_options(CoMD PRIVATE ${OPTFLAGS})

# Apply SYCL flags to C++ only
target_compile_options(CoMD PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${SYCL_FLAGS}>)

# If user provided raw MPI_INCLUDE with "-I..." flags, keep behavior similar by appending as options
# (This mirrors Makefile semantics more closely than CMake include dirs.)
if(DO_MPI AND NOT MPI_INCLUDE STREQUAL "")
  separate_arguments(_MPI_INCLUDE_LIST NATIVE_COMMAND "${MPI_INCLUDE}")
  target_compile_options(CoMD PRIVATE ${_MPI_INCLUDE_LIST})
endif()

# If user provided raw MPI_LIB flags, append at link
if(DO_MPI AND NOT MPI_LIB STREQUAL "")
  separate_arguments(_MPI_LIB_LIST NATIVE_COMMAND "${MPI_LIB}")
  target_link_options(CoMD PRIVATE ${_MPI_LIB_LIST})
endif()

# Math lib on UNIX to match Makefile
if(UNIX)
  target_link_libraries(CoMD PRIVATE m)
endif()

# Output name and directory (mirror Makefile)
set_target_properties(CoMD PROPERTIES
  OUTPUT_NAME "${CoMD_VARIANT}"
  RUNTIME_OUTPUT_DIRECTORY "${BIN_DIR}"
)