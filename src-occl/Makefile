# Makefile for SYCL version with oneCCL support

SHELL = /bin/bash

### Specify Intel oneAPI paths if necessary
ONEAPI_ROOT ?= /opt/intel/oneapi

### CoMD can be built in either double or single precision and with or
### without MPI.  Select desired precision and MPI here.

# double precision (ON/OFF)
DOUBLE_PRECISION = ON
# MPI for parallel (ON/OFF)
DO_MPI = ON
# oneCCL for collective operations (ON/OFF)
USE_ONECCL = ON
# Maximum number of atoms that can be stored in a link cell
MAXATOMS = 256

### Set your desired C++ compiler and any necessary flags.
### For SYCL we use icpx (Intel oneAPI DPC++ compiler)
CXX = icpx
CC = mpicc

# SYCL flags
SYCL_FLAGS = -fsycl -fsycl-targets=spir64
# For Intel GPUs specifically:
# SYCL_FLAGS = -fsycl -fsycl-targets=spir64_gen -Xs "-device pvc"

CXXFLAGS = -std=c++17 -DMPICH_SKIP_MPICXX -DOMPI_SKIP_MPICXX -DMAXATOMS=$(MAXATOMS) $(SYCL_FLAGS)
CFLAGS = -std=c99 -Wno-unused-result -DMAXATOMS=$(MAXATOMS)

OPTFLAGS = -g -O3
INCLUDES = 
C_LIB = -lm

### If you need to specify include paths, library paths, or link flags
### for MPI, put them here.  Put both -L and -l switches into MPI_LIB.
MPI_LIB = -lmpi
MPI_INCLUDE = 

### oneCCL paths
CCL_ROOT ?= $(ONEAPI_ROOT)/ccl/latest
CCL_INCLUDE = -I$(CCL_ROOT)/include
CCL_LIB = -L$(CCL_ROOT)/lib -lccl

### A place to specify any other include or library switches your
### platform requires.
OTHER_LIB = 
OTHER_INCLUDE = 

#########################################
### Below here, it is pitch black.  
### You are likely to be eaten by a grue.
##########################################

# clear all suffixes
.SUFFIXES:
# list only those that we use 
.SUFFIXES: .c .cpp .o

.PHONY: DEFAULT clean distclean depend

BIN_DIR=../bin

# Check for double precision
ifeq ($(DOUBLE_PRECISION), ON)
CFLAGS += -DCOMD_DOUBLE
CXXFLAGS += -DCOMD_DOUBLE
else
CFLAGS += -DCOMD_SINGLE
CXXFLAGS += -DCOMD_SINGLE
endif

# Set executable name and add includes & libraries for MPI if needed.
CoMD_VARIANT = CoMD-sycl
ifeq ($(DO_MPI), ON)
CoMD_VARIANT = CoMD-sycl-mpi
INCLUDES += ${MPI_INCLUDE}
CFLAGS += -DDO_MPI
CXXFLAGS += -DDO_MPI
LDFLAGS += ${MPI_LIB}
endif

# Add oneCCL support
ifeq ($(USE_ONECCL), ON)
INCLUDES += ${CCL_INCLUDE}
CXXFLAGS += -DUSE_ONECCL
LDFLAGS += ${CCL_LIB}
CoMD_VARIANT = CoMD-sycl-mpi-ccl
endif

CoMD_EXE = ${BIN_DIR}/${CoMD_VARIANT}

LDFLAGS += ${C_LIB} ${OTHER_LIB}
CFLAGS  += ${OPTFLAGS} ${INCLUDES} ${OTHER_INCLUDE}
CXXFLAGS += ${OPTFLAGS} ${INCLUDES} ${OTHER_INCLUDE}

# C files to be compiled with mpicc/gcc
C_SOURCES = cmdLineParser.c \
            decomposition.c \
            hashTable.c \
            initAtoms.c \
            ljForce.c \
            mycommand.c \
            mytype.c \
            neighborList.c \
            performanceTimers.c \
            random.c \
            yamlOutput.c

C_OBJECTS = $(C_SOURCES:.c=.o)

# C++ files to be compiled with icpx (SYCL)
CXX_SOURCES = CoMD.cpp \
              parallel.cpp \
              gpu_utility.cpp \
              gpu_kernels.cpp \
              haloExchange.cpp \
              linkCells.cpp \
              eam.cpp \
              timestep.cpp \
              gpu_neighborList.cpp

CXX_OBJECTS = $(CXX_SOURCES:.cpp=.o)

OBJECTS = $(C_OBJECTS) $(CXX_OBJECTS)

DEFAULT: ${CoMD_EXE}

%.o: %.c
	${CC} ${CFLAGS} -c $< -o $@

%.o: %.cpp
	${CXX} ${CXXFLAGS} -c $< -o $@

${CoMD_EXE}: ${BIN_DIR} CoMD_info.h ${OBJECTS}
	${CXX} ${CXXFLAGS} -o ${CoMD_EXE} ${OBJECTS} ${LDFLAGS}

CoMD_info.h: Makefile
	./generate_info_header ${CoMD_VARIANT} "$(CXX)" "$(CXXFLAGS)" "$(LDFLAGS)"

${BIN_DIR}:
	@if [ ! -d ${BIN_DIR} ]; then mkdir -p ${BIN_DIR} ; fi

clean:
	rm -f *.o CoMD_info.h .depend

distclean: clean
	rm -f ${CoMD_EXE} .depend.bak
	rm -rf html latex

depend:
	touch .depend
	makedepend -f .depend -Y. --$(CFLAGS)-- $(C_SOURCES) 2> /dev/null
	makedepend -a -f .depend -Y. --$(CXXFLAGS)-- $(CXX_SOURCES) 2> /dev/null

-include .depend
