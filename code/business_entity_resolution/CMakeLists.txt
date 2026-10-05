cmake_minimum_required(VERSION 3.15)
project(business_entity_resolution CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

find_package(Threads REQUIRED)

add_library(er_core STATIC
    src/normalize.cpp
    src/tsv_io.cpp
    src/blocking.cpp
    src/features.cpp
)
target_include_directories(er_core PUBLIC src)
target_link_libraries(er_core PUBLIC Threads::Threads)

add_executable(train_er src/train_main.cpp)
target_link_libraries(train_er PRIVATE er_core)

add_executable(predict_er src/predict_main.cpp)
target_link_libraries(predict_er PRIVATE er_core)
