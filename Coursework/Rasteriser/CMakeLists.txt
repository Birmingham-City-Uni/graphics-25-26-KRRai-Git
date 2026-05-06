cmake_minimum_required(VERSION 3.27.1)

project(Lab6)

add_subdirectory(3rdParty)

include_directories(3rdParty/lodepng)
include_directories(../../3rdParty/eigen-3.4.0)

add_executable(Task1
    Task1.cpp
    Mesh.hpp
    Light.hpp
    LinAlg.hpp
    Image.hpp
    )

target_link_libraries(Task1 
    lodepng
    )
