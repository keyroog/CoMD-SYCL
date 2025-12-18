@echo off
setlocal

REM 1) oneAPI env
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" || exit /b 1

REM 2) MSYS2 paths
set "PATH=C:\msys64\ucrt64\bin;C:\msys64\usr\bin;%PATH%"

REM 3) repo path (modifica se serve)
set "REPO=C:\Users\gamba\WSs\cWs\CoMD-SYCL"

REM 4) run build in bash -lc (SINGLE LINE)
C:\msys64\usr\bin\bash.exe -lc "set -e; cd '/c/Users/gamba/WSs/cWs/CoMD-SYCL/src-occl'; echo PWD=$(pwd); ls -la; which make; which icpx; icpx --version; make -j"