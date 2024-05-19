cmake_minimum_required(VERSION 2.8)
project(Projeto1)

find_package( OpenCV REQUIRED )

include_directories(include)

add_executable(projeto1
    main.cpp
    src/parallel.cpp
    src/sequential.cpp
    src/util.cpp
    include/defines.h
    include/parallel.h
    include/sequential.h
    include/util.h
)

target_link_libraries( projeto1 ${OpenCV_LIBS} )