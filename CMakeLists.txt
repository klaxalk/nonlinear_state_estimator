cmake_minimum_required(VERSION 2.8.3)
project(heading_estim_test)

find_package(catkin REQUIRED COMPONENTS
  cmake_modules
  roscpp
  sensor_msgs
  geometry_msgs
  std_msgs
  nav_msgs
  mrs_msgs
  mrs_lib
  message_generation
  actionlib
  nodelet
  tf
  dynamic_reconfigure
  )

generate_dynamic_reconfigure_options(
  cfg/heading_estim_test.cfg
  )

find_package(Eigen3 REQUIRED)
set(Eigen_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIRS})
set(Eigen_LIBRARIES ${EIGEN3_LIBRARIES})

catkin_package(
  CATKIN_DEPENDS mrs_msgs roscpp sensor_msgs std_msgs geometry_msgs mrs_lib
  DEPENDS Eigen
  LIBRARIES HeadingEstimTest
  )

find_package(Boost REQUIRED COMPONENTS thread)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${dynamic_reconfigure_PACKAGE_PATH}/cmake/cfgbuild.cmake
  )

# HeadingEstimTest

add_library(HeadingEstimTest src/heading_estim_test.cpp)

add_dependencies(HeadingEstimTest
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
  )

target_link_libraries(HeadingEstimTest
  ${catkin_LIBRARIES}
  ${EIGEN3_LIBRARIES}
  )
