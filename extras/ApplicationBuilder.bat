ECHO ON
CD ..
SET "buildType=%1"
RMDIR /q /s build
SET "SRC_DIR=%CD%"
SET "EIGEN3_DIR=D:\\vision\\eigen"
IF "%1"=="Release" (SET "TORCH_DIR=D:\\pytorch\\library\\Release\\libtorch") ELSE (SET "TORCH_DIR=D:\\pytorch\\library\\Debug\\libtorch") 
MKDIR build
CD build
SET "BUILD_DIR=%CD%"
SET "generator=Visual Studio 16 2019"
"C:\Program Files\CMake\bin\cmake.exe" -B"%BUILD_DIR%" -H"%SRC_DIR%" -G"%generator%" -DEIGEN3_DIR=%EIGEN3_DIR% -DTORCH_DIR=%TORCH_DIR%
"C:\Program Files\CMake\bin\cmake.exe" --build %BUILD_DIR% --config %buildType% --target lanedetection
COPY %SRC_DIR%\best.torchscript.pt %BUILD_DIR%\%buildType%
COPY %SRC_DIR%\opencv_resized.mp4 %BUILD_DIR%\%buildType%
COPY %SRC_DIR%\calibration.yml %BUILD_DIR%\%buildType%
CD ../extras