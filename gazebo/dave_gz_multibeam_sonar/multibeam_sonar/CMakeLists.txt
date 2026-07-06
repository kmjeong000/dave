cmake_minimum_required(VERSION 3.11.0 FATAL_ERROR)

# Suppress developer warnings for the entire workspace.
set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS TRUE CACHE BOOL
    "Suppress developer warnings" FORCE)
# Set CMP0144 to NEW to ensure find_package
# uses upper-case <PACKAGENAME>_ROOT variables.
if(POLICY CMP0144)
  cmake_policy(SET CMP0144 NEW)
endif()

project(multibeam_sonar)

find_package(ament_cmake REQUIRED)
find_package(CUDAToolkit QUIET)

if(CUDAToolkit_FOUND)
  enable_language(CUDA)
  find_package(CUDA REQUIRED)
  message(STATUS "CUDA found, enabling CUDA support.")
  include_directories(${CUDA_INCLUDE_DIRS})
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_60")

  find_package(gz-cmake3 REQUIRED)
  find_package(gz-sim8 REQUIRED)
  find_package(gz-sensors8 REQUIRED)
  find_package(gz-rendering8 REQUIRED OPTIONAL_COMPONENTS ogre ogre2)
  find_package(gz-transport13 REQUIRED)
  find_package(rclcpp REQUIRED)
  find_package(sensor_msgs REQUIRED)
  find_package(geometry_msgs REQUIRED)
  find_package(rosidl_default_generators REQUIRED)
  find_package(PCL REQUIRED)
  find_package(pcl_conversions REQUIRED)
  find_package(OpenCV REQUIRED)
  find_package(marine_acoustic_msgs REQUIRED)
  find_package(cv_bridge REQUIRED)

  include_directories(${PCL_INCLUDE_DIRS})
  link_directories(${PCL_LIBRARY_DIRS})
  add_definitions(${PCL_DEFINITIONS})

  # Set version variables
  set(GZ_MSGS_VER ${gz-msgs10_VERSION_MAJOR})
  set(GZ_RENDERING_VER ${gz-rendering8_VERSION_MAJOR})
  set(GZ_SENSORS_VER ${gz-sensors8_VERSION_MAJOR})
  set(GZ_TRANSPORT_VER ${gz-transport13_VERSION_MAJOR})
  set(GZ_SIM_VER ${gz-sim8_VERSION_MAJOR})

  if(TARGET gz-rendering${GZ_RENDERING_VER}::ogre)
    set(HAVE_OGRE TRUE)
    set(GZ_RENDERING_TARGET gz-rendering${GZ_RENDERING_VER}-ogre)
    add_definitions(-DWITH_OGRE)
  endif()

  if(TARGET gz-rendering${GZ_RENDERING_VER}::ogre2)
    set(HAVE_OGRE2 TRUE)
    set(GZ_RENDERING_TARGET gz-rendering${GZ_RENDERING_VER}-ogre2)
    add_definitions(-DWITH_OGRE2)
  endif()

  add_library(${PROJECT_NAME}
    SHARED
      MultibeamSonarSensor.cc
      sonar_calculation_cuda.cu
  )

  set_target_properties(${PROJECT_NAME}
                          PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

  target_include_directories(${PROJECT_NAME}
    PUBLIC
    ${gz-sensors${GZ_SENSORS_VER}_INCLUDE_DIRS}
    ${gz-rendering${GZ_RENDERING_VER}_INCLUDE_DIRS}
    ${gz-msgs${GZ_MSGS_VER}_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${gz-transport${GZ_TRANSPORT_VER}_INCLUDE_DIRS}
    ${gz-sim${GZ_SIM_VER}_INCLUDE_DIRS}
  )

  target_link_libraries(${PROJECT_NAME}
  ${PCL_LIBRARIES}
  ${CUDA_LIBRARIES}
  ${CUDA_CUFFT_LIBRARIES}
  ${CUBLAS_LIB})

  install(TARGETS ${PROJECT_NAME}
    DESTINATION lib/${PROJECT_NAME}
  )

  ament_target_dependencies(${PROJECT_NAME}
    gz-sensors${GZ_SENSORS_VER}
    ${GZ_RENDERING_TARGET}
    gz-msgs${GZ_MSGS_VER}
    gz-transport${GZ_TRANSPORT_VER}
    gz-sim${GZ_SIM_VER}
    rclcpp
    sensor_msgs
    rosidl_default_generators
    PCL
    pcl_conversions
    OpenCV
    marine_acoustic_msgs
    geometry_msgs
    cv_bridge
  )


  # Environment hooks
  ament_environment_hooks(
    "${CMAKE_CURRENT_SOURCE_DIR}/hooks/${PROJECT_NAME}.dsv.in"
  )

else()
  message(STATUS "CUDA Toolkit not found: Skipping CUDA-specific targets")
endif()

ament_package()
