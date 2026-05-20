cmake_minimum_required(VERSION 3.16)

set(EXTRA_COMPONENT_DIRS "$ENV{HOME}/esp/esp-who/components")
include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(driver_monitoring_system)