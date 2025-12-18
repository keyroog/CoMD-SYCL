@echo off
REM 1) oneAPI environment (x64)
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" || exit /b 1

REM 2) Prepend MSYS2 (UCRT64) paths
set "PATH=C:\msys64\ucrt64\bin;C:\msys64\usr\bin;%PATH%"

REM 3) Build (bash login shell)
C:\msys64\usr\bin\bash.exe -lc ^
  "cd /c/Users/gamba/WSs/cWs/CoMD-SYCL/src-occl && \
   uname -a && \
   which make && which bash && \
   which icpx || true && \
   which mpicc || true && \
   icpx --version && \
   make -j"