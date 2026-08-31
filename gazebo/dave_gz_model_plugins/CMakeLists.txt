cmake_minimum_required(VERSION 3.8)
project(dave_gz_model_plugins)

# Find required packages
find_package(ament_cmake REQUIRED)
find_package(Boost REQUIRED COMPONENTS system)
find_package(gz-cmake3 REQUIRED)
find_package(gz-plugin2 REQUIRED COMPONENTS register)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(gz-common5 REQUIRED COMPONENTS profiler)
find_package(gz-sim8 REQUIRED)
find_package(gz-math7 REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(dave_interfaces REQUIRED)
find_package(Protobuf REQUIRED)
find_package(gz-msgs10 REQUIRED)
find_package(gz-transport13 REQUIRED)
find_package(dave_gz_world_plugins REQUIRED)

# Set version variables
set(GZ_PLUGIN_VER ${gz-plugin2_VERSION_MAJOR})
set(GZ_COMMON_VER ${gz-common5_VERSION_MAJOR})
set(GZ_SIM_VER ${gz-sim8_VERSION_MAJOR})
set(GZ_MATH_VER ${gz-math7_VERSION_MAJOR})
set(GZ_MSGS_VER ${gz-msgs10_VERSION_MAJOR})
set(GZ_TRANSPORT_VER ${gz-transport13_VERSION_MAJOR})

message(STATUS "Compiling against Gazebo Harmonic")

add_library(OceanCurrentModelPlugin SHARED src/OceanCurrentModelPlugin.cc)

target_include_directories(OceanCurrentModelPlugin PRIVATE include)

target_link_libraries(OceanCurrentModelPlugin
  gz-sim${GZ_SIM_VER}::gz-sim${GZ_SIM_VER}
)

# Specify dependencies for ocean_current_plugin using ament_target_dependencies
ament_target_dependencies(OceanCurrentModelPlugin
  rclcpp
  std_msgs
  geometry_msgs
  dave_interfaces
  dave_gz_world_plugins
)

add_library(SailLiftDragSystem SHARED
  src/SailLiftDragSystem.cc
  src/LiftDragModel.cc
)

target_include_directories(SailLiftDragSystem PRIVATE 
  include
)

target_link_libraries(SailLiftDragSystem
  gz-sim${GZ_SIM_VER}::gz-sim${GZ_SIM_VER}
  gz-plugin${GZ_PLUGIN_VER}::register
  gz-math${GZ_MATH_VER}::gz-math${GZ_MATH_VER}
  gz-msgs${GZ_MSGS_VER}::gz-msgs${GZ_MSGS_VER}
  gz-transport${GZ_TRANSPORT_VER}::gz-transport${GZ_TRANSPORT_VER}
)

add_library(FoilLiftDragSystem SHARED
  src/FoilLiftDragSystem.cc
  src/LiftDragModel.cc
)

target_include_directories(FoilLiftDragSystem PRIVATE 
  include
)

target_link_libraries(FoilLiftDragSystem
  gz-sim${GZ_SIM_VER}::gz-sim${GZ_SIM_VER}
  gz-plugin${GZ_PLUGIN_VER}::register
  gz-math${GZ_MATH_VER}::gz-math${GZ_MATH_VER}
  gz-msgs${GZ_MSGS_VER}::gz-msgs${GZ_MSGS_VER}
  gz-transport${GZ_TRANSPORT_VER}::gz-transport${GZ_TRANSPORT_VER}
)

# Install targets
install(
  TARGETS 
    OceanCurrentModelPlugin
    SailLiftDragSystem
    FoilLiftDragSystem
  DESTINATION lib/${PROJECT_NAME}
)

# Install headers
install(
  DIRECTORY include/
  DESTINATION include/
)

# Environment hooks
ament_environment_hooks(
  "${CMAKE_CURRENT_SOURCE_DIR}/hooks/${PROJECT_NAME}.dsv.in"
)

# Testing setup
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
