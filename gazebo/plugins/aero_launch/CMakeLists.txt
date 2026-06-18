cmake_minimum_required(VERSION 3.16)
project(ttsim_aero_launch)

# Target Gazebo Harmonic (gz-sim8). For Garden use gz-sim7; for Ionic gz-sim9.
find_package(gz-cmake3 REQUIRED)
find_package(gz-sim8 REQUIRED)
find_package(gz-plugin2 REQUIRED COMPONENTS register)
find_package(gz-math7 REQUIRED)

add_library(AeroLaunch SHARED AeroLaunch.cc)
target_include_directories(AeroLaunch PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(AeroLaunch
  PRIVATE
    gz-sim8::gz-sim8
    gz-plugin2::register
    gz-math7::gz-math7
)

# Build, then add this dir to GZ_SIM_SYSTEM_PLUGIN_PATH:
#   cmake -B build && cmake --build build
#   export GZ_SIM_SYSTEM_PLUGIN_PATH=$PWD/build:$GZ_SIM_SYSTEM_PLUGIN_PATH
