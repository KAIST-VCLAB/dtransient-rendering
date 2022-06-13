cmake_minimum_required(VERSION 3.12)

project(dtrr VERSION 1.0.1 DESCRIPTION "Differentiable Transient Renderer")


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

if(WIN32)
    find_package(PythonLibs 3.6 COMPONENTS Development REQUIRED)
    add_subdirectory(pybind11)
elseif(APPLE)
    find_package(Python 3.6 COMPONENTS Development REQUIRED)
    include_directories(${PYBIND_INCLUDES})
else()
    find_package(Python 3.6 COMPONENTS Development REQUIRED)
endif()

find_package(Embree REQUIRED)
if(NOT EMBREE_FOUND)
    #add_subdirectory(embree)
    add_subdirectory(dependencies/embree)
endif()

# Find OpenMP
if(APPLE)
    if(CMAKE_C_COMPILER_ID MATCHES "Clang")
        set(OpenMP_C "${CMAKE_C_COMPILER}")
        set(OpenMP_C_FLAGS "-fopenmp=libomp -Wno-unused-command-line-argument")
        set(OpenMP_C_LIB_NAMES "libomp" "libgomp" "libiomp5")
        set(OpenMP_libomp_LIBRARY ${OpenMP_C_LIB_NAMES})
        set(OpenMP_libgomp_LIBRARY ${OpenMP_C_LIB_NAMES})
        set(OpenMP_libiomp5_LIBRARY ${OpenMP_C_LIB_NAMES})
    endif()
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
      set(OpenMP_CXX "${CMAKE_CXX_COMPILER}")
      set(OpenMP_CXX_FLAGS "-fopenmp=libomp -Wno-unused-command-line-argument")
      set(OpenMP_CXX_LIB_NAMES "libomp" "libgomp" "libiomp5")
      set(OpenMP_libomp_LIBRARY ${OpenMP_CXX_LIB_NAMES})
      set(OpenMP_libgomp_LIBRARY ${OpenMP_CXX_LIB_NAMES})
      set(OpenMP_libiomp5_LIBRARY ${OpenMP_CXX_LIB_NAMES})
    endif()
    include_directories("${OPENMP_INCLUDES}")
    link_directories("${OPENMP_LIBRARIES}")
endif()

find_package(OpenMP REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

include_directories(${Python_INCLUDE_DIRS})
include_directories(${EMBREE_INCLUDE_PATH})
include_directories(include/)

if (MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX2 /O2 /Z7 /wd4244 /wd4267 /wd4305 /wd4996")
elseif (APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -Wno-ignored-attributes -Wno-macro-redefined -Wall -g -O3 -fvisibility=hidden")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -Wno-ignored-attributes -Wall -g -O3 -fvisibility=hidden")
endif()

set(SRCS include/config.h
         include/fwd.h
         include/ptr.h
         include/stats.h
         src/stats.cpp
         include/math_func.h
         src/math_func.cpp
         include/line_clip.h
         include/utils.h
         src/utils.cpp
         include/frame.h
         include/frameAD.h
         include/ray.h
         include/rayAD.h
         include/sampler.h
         src/sampler.cpp
         include/camera.h
         src/camera.cpp
         include/phase.h
         include/shape.h
         src/shape.cpp
         include/intersection.h
         include/intersectionAD.h
         include/edge_manager.h
         src/edge_manager/edge_manager.cpp
         src/edge_manager/bruteforce.h
         src/edge_manager/bruteforce.cpp
         src/edge_manager/tree.h
         src/edge_manager/tree.cpp
         src/edge_manager/pathspace.h
         src/edge_manager/pathspace.cpp
         include/scene.h
         src/scene.cpp
         include/emitter.h
         src/emitter/area.h
         src/emitter/area.cpp
         src/emitter/area2.h
         src/emitter/area2.cpp
         include/bsdf.h
         src/bsdf.cpp
         src/bsdf/null.h
         src/bsdf/null.cpp
         src/bsdf/diffuse.h
         src/bsdf/diffuse.cpp
         src/bsdf/texturedDiffuse.h
         src/bsdf/texturedDiffuse.cpp
         src/bsdf/phong.h
         src/bsdf/phong.cpp
         src/bsdf/microfacet.h
         src/bsdf/roughconductor.h
         src/bsdf/roughconductor.cpp
         src/bsdf/roughdielectric.h
         src/bsdf/roughdielectric.cpp
         src/bsdf/twosided.h
         src/bsdf/twosided.cpp
         include/medium.h
         src/medium/homogeneous.h
         src/medium/homogeneous.cpp
         src/medium/gridvolume.h
         src/medium/heterogeneous.h
         src/medium/heterogeneous.cpp
         include/integrator.h
         include/integratorAD.h
         src/integrator/differential/integratorAD.cpp
         include/integratorADps.h
         src/integrator/differential/integratorADps.cpp
         src/integrator/differential/bidir_utils.h
         src/integrator/differential/bidir_utils.cpp
         src/integrator/differential/bdptAD.h
         src/integrator/differential/bdptAD.cpp
         src/integrator/bdpt.h
         src/integrator/bdpt.cpp
         include/cameratransient.h src/cameratransient.cpp src/integrator/differential/bdptADbb.h src/integrator/differential/bdptADbb.cpp)

set(SRCS_BIND src/dtrr.cpp)
#set(SRCS_MAIN src/main.cpp)

if (APPLE)
    set(DYNAMIC_LOOKUP "-undefined dynamic_lookup")
endif()
if (WIN32)
    pybind11_add_module(dtrr SHARED ${SRCS} ${SRCS_BIND})
endif()

if (NOT WIN32)
    add_library(dtrr MODULE ${SRCS} ${SRCS_BIND})
    # The "-undefined dynamic_lookup" is a hack for systems with
    # multiple Python installed. If we link a particular Python version
    # here, and we import it with a different Python version later.
    # likely a segmentation fault.
    # The solution for Linux/Mac OS machines, as mentioned in
    # https://github.com/pybind/pybind11/blob/master/tools/pybind11Tools.cmake
    # is to not link against Python library at all and resolve the symbols
    # at compile time.
    target_link_libraries(dtrr
        ${EMBREE_LIBRARY}
        ${DYNAMIC_LOOKUP})
else()
    target_link_libraries(dtrr
    PRIVATE
        ${EMBREE_LIBRARY}
        )
endif()

set_target_properties(dtrr PROPERTIES PREFIX "")

if (WIN32)
    set(PYTHON_EXEC "python")
else()
    set(PYTHON_EXEC "python3")
endif()

if(DEFINED _NDER)
    add_definitions(-DNDER=${_NDER})
    message("##############################")
    message( "-- NDER has set to be ${_NDER}. Is this the value what you want?")
    message("##############################")
else()
    add_definitions(-DNDER=1)
    message( "-- NDER has set to be 1")

endif()


execute_process(
    COMMAND ${PYTHON_EXEC} -c "if True:
        from distutils import sysconfig as sc
        print(sc.get_python_lib())"
    OUTPUT_VARIABLE PYTHON_SITE
    OUTPUT_STRIP_TRAILING_WHITESPACE)

install(
    TARGETS dtrr
    LIBRARY DESTINATION lib
    DESTINATION ${PYTHON_SITE})

execute_process (
    COMMAND ${PYTHON_EXEC} -c "from distutils.sysconfig import get_python_version; print(get_python_version())"
    OUTPUT_VARIABLE PYTHON_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE)
install(CODE "execute_process(COMMAND ${PYTHON_EXEC} ${CMAKE_CURRENT_SOURCE_DIR}/setup.py install
                              WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})")
                              
# Main executable for C++ run
#add_executable (dtrr.out ${SRCS} ${SRCS_MAIN})
#target_link_libraries(dtrr.out ${EMBREE_LIBRARY} ${DYNAMIC_LOOKUP})
