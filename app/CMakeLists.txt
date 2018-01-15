cmake_minimum_required(VERSION 3.4.1)

set(CMAKE_VERBOSE_MAKEFILE on)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")

add_library(
            lib_opencv
            SHARED
            IMPORTED
            )
set_target_properties(
    lib_opencv
    PROPERTIES
    IMPORTED_LOCATION
    ${CMAKE_CURRENT_LIST_DIR}/src/main/jniLibs/${ANDROID_ABI}/libopencv_java3.so
    )

include(AndroidNdkModules)
android_ndk_import_module_cpufeatures()

add_library(
    caffe2
    STATIC
    IMPORTED
    )
set_target_properties(
    caffe2
    PROPERTIES IMPORTED_LOCATION
    ${CMAKE_CURRENT_LIST_DIR}/src/main/jniLibs/${ANDROID_ABI}/libcaffe2.a
    )

add_library(
    thread_pool
    STATIC
    IMPORTED
    )
set_target_properties(
    thread_pool
    PROPERTIES IMPORTED_LOCATION
    ${CMAKE_CURRENT_LIST_DIR}/src/main/jniLibs/${ANDROID_ABI}/libpthreadpool.a
    )

add_library(
    protobuf
    SHARED
    IMPORTED
    )
set_target_properties(
    protobuf
    PROPERTIES IMPORTED_LOCATION
    ${CMAKE_CURRENT_LIST_DIR}/src/main/jniLibs/${ANDROID_ABI}/libprotobuf.a
    )

add_library(
         NNPACK
         STATIC
         IMPORTED
         )
     set_target_properties(
         NNPACK
         PROPERTIES IMPORTED_LOCATION
         ${CMAKE_CURRENT_LIST_DIR}/src/main/jniLibs/${ANDROID_ABI}/libnnpack.a
         )

add_library(
    glog
    SHARED
    IMPORTED
    )
set_target_properties(
    glog
    PROPERTIES IMPORTED_LOCATION
    ${CMAKE_CURRENT_LIST_DIR}/src/main/jniLibs/${ANDROID_ABI}/libglog.so
    )

include_directories( src/main/cpp )

add_library(
    native-lib
    SHARED
    src/main/cpp/native-lib.cpp
)

find_library( log-lib log )
find_library( android-lib android)

target_link_libraries(
                       native-lib
                       lib_opencv
                       EGL
                       GLESv2
                       -Wl,--whole-archive
                       caffe2
                       -Wl,--no-whole-archive
                       NNPACK
                       thread_pool
                       glog
                       protobuf
                       cpufeatures
                       ${log-lib}
                       ${android-lib})