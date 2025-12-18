@echo off
setlocal

REM 1) Forza setvars (così non “skip”)
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" --force || exit /b 1

REM 2) Fai ereditare a MSYS2 il PATH di Windows (incluso oneAPI)
set "MSYS2_PATH_TYPE=inherit"

REM 3) Metti MSYS2 davanti (così trovi make/bash/coreutils)
set "PATH=C:\msys64\ucrt64\bin;C:\msys64\usr\bin;%PATH%"

REM 4) Esegui build (una riga, niente backslash)
C:\msys64\usr\bin\bash.exe -lc "set -e; cd '/c/Users/gamba/WSs/cWs/CoMD-SYCL/src-occl'; echo PWD=$(pwd); which make; which icpx; icpx --version; make -j"