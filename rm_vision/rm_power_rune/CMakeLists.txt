cmake_minimum_required(VERSION 3.5)
set(CMAKE_CXX_STANDARD 17)
project(power_rune)

## Use C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

## By adding -Wall and -Werror, the compiler does not ignore warnings anymore,
## enforcing cleaner code.
add_definitions(-Wall -Werror)

## Export compile commands for clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#######################
## Find dependencies ##
#######################

find_package(ament_cmake_auto REQUIRED)
find_package(OpenCV REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(Ceres REQUIRED)
find_package(REQUIRED COMPONENTS
  rclcpp
  sensor_msgs)
find_package(global_interface REQUIRED)

ament_auto_find_build_dependencies()



###########
## Build ##
###########

ament_auto_add_library(${PROJECT_NAME} SHARED
  DIRECTORY src
)

target_include_directories(
    ${PROJECT_NAME} 
    PUBLIC
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
)

target_link_libraries(
    ${PROJECT_NAME}  
    ${OpenCV_LIBS}
    ${CERES_LIBRARIES}
)
            

rclcpp_components_register_node(${PROJECT_NAME}
  PLUGIN power_rune::PowerRuneNode
  EXECUTABLE power_rune_node
)


#############
## Install ##
#############

ament_auto_package(
  INSTALL_TO_SHARE
  config
)