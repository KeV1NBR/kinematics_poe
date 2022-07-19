# Project Dependences Configuration

# Backup and set build type to release
if(NOT MSVC)
    set(CMAKE_BUILD_TYPE_BAK ${CMAKE_BUILD_TYPE})
    set(CMAKE_BUILD_TYPE Release)
endif()

# Include subdirectories
include_directories(${DEPS_PATHS})
set(CMAKE_PREFIX_PATH "$ENV{HOME}/api/libtorch")

find_package(Torch REQUIRED)
include_directories(${TORCH_INCLUDE_DIR})
include_directories(${TORCH_INCLUDE_DIR})

# Add subdirectory
foreach(DEPS_PATH ${DEPS_PATHS})
    add_subdirectory(${DEPS_PATH})
endforeach()

# Restore origin build type
if(NOT MSVC)
    set(CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE_BAK})
endif()

