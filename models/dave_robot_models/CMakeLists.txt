cmake_minimum_required(VERSION 3.8)
project(dave_robot_models)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY description launch meshes config
  DESTINATION share/dave_robot_models
)

install(
  PROGRAMS
    scripts/command_adapter_core.py
    scripts/sailboat_command_adapter.py
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(
    command_adapter_core_test
    test/test_command_adapter_core.py
  )
endif()

ament_environment_hooks(
  "${CMAKE_CURRENT_SOURCE_DIR}/hooks/${PROJECT_NAME}.dsv.in")

ament_package()
