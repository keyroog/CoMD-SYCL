# Makefile for SYCL version with oneCCL support (portable Linux/Windows)

# -----------------------------
# OS-specific defaults & helpers
# -----------------------------
ifeq ($(OS),Windows_NT)
  # Do NOT force /bin/bash on Windows; use cmd.exe for recipes
  SHELL := cmd.exe
  .SHELLFLAGS := /c

  # oneAPI default root (Windows)
  ONEAPI_ROOT ?= C:/Program Files (x86)/Intel/oneAPI

  # Windows command helpers
  NULLDEV := NUL
  RM_F    := del /F /Q
  RMDIR_F := rmdir /S /Q
  MKDIR_P := if not exist "$(1)" mkdir "$(1)"
else
  # Linux/macOS
  SHELL = /bin/bash

  ONEAPI_ROOT ?= /opt/intel/oneapi

  NULLDEV := /dev/null
  RM_F    := rm -f
  RMDIR_F := rm -rf
  MKDIR_P := mkdir -p
endif

# -----------------------------
# Build configuration (same knobs as original)
# -----------------------------
DOUBLE_PRECISION = ON
DO_MPI = ON
USE_ONECCL = ON
MAXATOMS = 256

# Compilers (keep as in original; override from env if needed)
CXX = icpx
CC  = mpicc

# SYCL flags
SYCL_FLAGS = -fsycl -fsycl-targets=spir64
# For Intel GPUs specifically:
# SYCL_FLAGS = -fsycl -fsycl-targets=spir64_gen -Xs "-device pvc"

CXXFLAGS = -std=c++17 -DMAXATOMS=$(MAXATOMS) $(SYCL_FLAGS)
CFLAGS   = -std=c99 -Wno-unused-result -DMAXATOMS=$(MAXATOMS)

OPTFLAGS = -g -O3
INCLUDES =

# libm only on Unix
ifeq ($(OS),Windows_NT)
  C_LIB =
else
  C_LIB = -lm
endif

MPI_LIB =
MPI_INCLUDE =

# oneCCL paths
CCL_ROOT ?= $(ONEAPI_ROOT)/ccl/latest
CCL_INCLUDE = -I$(CCL_ROOT)/include

# Library dir differs sometimes; keep lib by default and allow override
CCL_LIBDIR ?= $(CCL_ROOT)/lib
CCL_LIB = -L$(CCL_LIBDIR) -lccl

OTHER_LIB =
OTHER_INCLUDE =

# -----------------------------
# Internals
# -----------------------------
.SUFFIXES:
.SUFFIXES: .c .cpp .o

.PHONY: DEFAULT clean distclean depend

BIN_DIR=../bin

# Precision
ifeq ($(DOUBLE_PRECISION), ON)
  CFLAGS   += -DCOMD_DOUBLE
  CXXFLAGS += -DCOMD_DOUBLE
else
  CFLAGS   += -DCOMD_SINGLE
  CXXFLAGS += -DCOMD_SINGLE
endif

# Variant & MPI
CoMD_VARIANT = CoMD-sycl
ifeq ($(DO_MPI), ON)
  CoMD_VARIANT = CoMD-sycl-mpi
  INCLUDES += $(MPI_INCLUDE)
  CFLAGS   += -DDO_MPI
  CXXFLAGS += -DDO_MPI
  LDFLAGS  += $(MPI_LIB)
endif

# oneCCL (same behavior: when ON -> mpi-ccl name)
ifeq ($(USE_ONECCL), ON)
  INCLUDES += $(CCL_INCLUDE)
  CXXFLAGS += -DUSE_ONECCL
  LDFLAGS  += $(CCL_LIB)
  CoMD_VARIANT = CoMD-sycl-mpi-ccl
endif

CoMD_EXE = $(BIN_DIR)/$(CoMD_VARIANT)

LDFLAGS  += $(C_LIB) $(OTHER_LIB)
CFLAGS   += $(OPTFLAGS) $(INCLUDES) $(OTHER_INCLUDE)
CXXFLAGS += $(OPTFLAGS) $(INCLUDES) $(OTHER_INCLUDE)

# Sources
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

DEFAULT: $(CoMD_EXE)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(CoMD_EXE): $(BIN_DIR) CoMD_info.h $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $(CoMD_EXE) $(OBJECTS) $(LDFLAGS)

# Header generation: same generator as Makefile
CoMD_info.h: Makefile
ifeq ($(OS),Windows_NT)
	REM If generate_info_header is a bash script, you must run this Makefile under a bash-capable environment.
	.\generate_info_header $(CoMD_VARIANT) "$(CXX)" "$(CXXFLAGS)" "$(LDFLAGS)"
else
	./generate_info_header $(CoMD_VARIANT) "$(CXX)" "$(CXXFLAGS)" "$(LDFLAGS)"
endif

$(BIN_DIR):
ifeq ($(OS),Windows_NT)
	@if not exist "$(BIN_DIR)" mkdir "$(BIN_DIR)"
else
	@mkdir -p "$(BIN_DIR)"
endif

clean:
ifeq ($(OS),Windows_NT)
	-@$(RM_F) *.o CoMD_info.h .depend 2>$(NULLDEV)
else
	-@$(RM_F) *.o CoMD_info.h .depend
endif

distclean: clean
ifeq ($(OS),Windows_NT)
	-@$(RM_F) "$(CoMD_EXE)" .depend.bak 2>$(NULLDEV)
	-@if exist html $(RMDIR_F) html
	-@if exist latex $(RMDIR_F) latex
else
	-@$(RM_F) $(CoMD_EXE) .depend.bak
	-@$(RMDIR_F) html latex
endif

# Optional dependency generation (often not available on Windows)
depend:
ifeq ($(OS),Windows_NT)
	@echo "depend: skipped on Windows (makedepend not configured)"
else
	touch .depend
	makedepend -f .depend -Y. --$(CFLAGS)-- $(C_SOURCES) 2> /dev/null
	makedepend -a -f .depend -Y. --$(CXXFLAGS)-- $(CXX_SOURCES) 2> /dev/null
endif

-include .depend