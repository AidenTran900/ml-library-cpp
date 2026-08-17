#
# Main CMakelists for the boilerplate project.
#
# It aims to be a template and a CMake reference, and as such is documented as much as possible.
# While everything might not fit in any project, it should give good defaults and avoid CMake antipatterns.
# If you disagree with some pieces of advice given here, please discuss it with me by opening a Github Issue !
#
# Project specific options :
#   - BP_USE_DOXYGEN
#   - BP_BUILD_TESTS (requires BUILD_TESTING set to ON)
# Other options might be available through the cmake scripts including (not exhaustive):
#   - ENABLE_WARNINGS_SETTINGS
#   - ENABLE_LTO
#

# Tells cmake the minimum version supported is 3.16, but has been updated to work with 3.29 policies
# It is recommended to use the `cmake_minimum_required(VERSION <min>[...<max>])` syntax to indicate the max version that was known to work.
# Many projects broke when CMake 4.0 was released even though they should have been compatible.
cmake_minimum_required(VERSION 3.16...3.29)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
    message(FATAL_ERROR "Do not build in-source. Please remove CMakeCache.txt and the CMakeFiles/ directory. Then build out-of-source.")
endif()

# Put the project early since modules might need to detect the compiler.
# More information https://cmake.org/cmake/help/latest/command/project.html
project(
    "Build" # This will exposed as the variable PROJECT_NAME.
    VERSION 0.1.0 # Used for installation and defines variables PROJECT_VERSION_MAJOR, PROJECT_VERSION_MINOR, PROJECT_VERSION_PATCH, and PROJECT_VERSION_TWEAK.
    LANGUAGES C CXX # Used to determine the languages to use based on file extensions
)

###############s
##   CUDA    ##
###############

option(ML_USE_CUDA "Enable CUDA GPU inference" OFF)

if(ML_USE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

############################
##  Modules and scripts   ##
############################

# Standard CMake modules

include(CTest)                          # Must be called before adding tests but after calling project(). This automatically calls enable_testing() and configures ctest targets when using Make/Ninja
include(CMakeDependentOption)           # This is a really useful scripts that creates options that depends on other options. It can even be used with generator expressions !
include(GNUInstallDirs)                 # This will define the default values for installation directories (all platforms even if named GNU)
include(InstallRequiredSystemLibraries) # Tell CMake that the `install` target needs to install required system libraries (eg: Windows SDK)
include(CMakePackageConfigHelpers)      # Helper to create relocatable packages

# Custom modules and scripts

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake") # Make our cmake scripts available

include(LTO)
include(Warnings)
include(CopyDllsForDebug)
include(Coverage)

###############
##  OPTIONS  ##
###############

# You should try to give as much control over the project setup to the user.
# When modifying compile flags for example, if they are not mandatory, provide an option.

option(${PROJECT_NAME}_USE_DOXYGEN "Add a doxygen target to generate the documentation" ON)
option(${PROJECT_NAME}_USE_ADDITIONAL_SOURCEFILE "Use the additional source file" ON)
option(${PROJECT_NAME}_INSTALL "Should ${PROJECT_NAME} be added to the install list? Useful if included using add_subdirectory." ON)

# Use your own option for tests, in case people use your library through add_subdirectory
cmake_dependent_option(${PROJECT_NAME}_BUILD_TESTS
    "Enable ${PROJECT_NAME} project tests targets" ON # By default we want tests if CTest is enabled
    "BUILD_TESTING" OFF # Stay coherent with CTest variables
)

# External dependencies
add_subdirectory(external EXCLUDE_FROM_ALL)

# It is always easier to navigate in an IDE when projects are organized in folders.
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Whe building a shared library, you do not want to export all symbols by default
# gcc (and hence clang) are wrong about this.
#
# For more information, see https://gcc.gnu.org/wiki/Visibility and https://www.youtube.com/embed/m0DwB4OvDXk
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN 1)

###############
##  OpenMP   ##
###############

find_package(OpenMP)

###############
##  Project  ##
###############

# set(CSV_CXX_STANDARD 11)
add_subdirectory(csv-parser)

# Check for LTO support (needs to be after project(...) )
find_lto(CXX)

#==========================#
#  Build executable  #
#==========================#

# Always list the source files explicitly, including headers so that they are listed in the IDE
# If you need to use files based on a variable value, use target_sources
add_executable(Build
    source/main.cpp
    source/math/matrix.cpp
    source/core/loss.cpp
    source/core/optimizer.cpp
    source/core/regularizer.cpp
    source/core/metrics.cpp
    source/models/linear-regression.cpp
    source/models/logistic-regression.cpp
    source/models/support-vector-machine.cpp
    source/utils/csv_utils.cpp
    source/utils/gguf-loader.cpp
    source/utils/llama-loader.cpp
    source/core/tokenizer.cpp
    examples/c++/linear-regression/housing/housing-example.cpp
    examples/logistic-regression/h-disease-example.cpp
    tests/math/matrix-test.cpp
    tests/models/linear-regression-test.cpp
    include/ml_lib/math/matrix.h
    include/ml_lib/core/loss.h
    include/ml_lib/core/optimizer.h
    include/ml_lib/core/regularizer.h
    include/ml_lib/core/metrics.h
    include/ml_lib/core/tokenizer.h
    include/ml_lib/core/token-sampler.h
    include/ml_lib/models/linear-regression.h
    include/ml_lib/models/logistic-regression.h
    include/ml_lib/models/support-vector-machine.h
    include/ml_lib/utils/csv_utils.h
    include/ml_lib/utils/gguf-loader.h
    include/ml_lib/utils/llama-loader.h
    examples/c++/linear-regression/housing/housing-example.h
    examples/logistic-regression/h-disease-example.h
)

target_include_directories(Build
    PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/source
        ${CMAKE_CURRENT_LIST_DIR}/include
        ${CMAKE_CURRENT_LIST_DIR}/tests
        ${CMAKE_CURRENT_LIST_DIR}/examples/c++/linear-regression/housing
        ${CMAKE_CURRENT_LIST_DIR}/examples/logistic-regression
)

target_link_libraries(Build
    #PUBLIC # Useful for libraries, see https://cmake.org/cmake/help/latest/manual/cmake-buildsystem.7.html for more details about transitive usage requirements.
        #libraries/targets to link when linking this library
        #this will automatically setup the needed flags and dependencies when linking against this target
    PRIVATE
        general fmt::fmt spdlog::spdlog csv $<$<BOOL:${OpenMP_CXX_FOUND}>:OpenMP::OpenMP_CXX>
        #debug DEBUGLIBS
        #optimized RELEASELIBS
)

# If you need platform specific files to be compiled and do not want to add a new target for this,
# you can use target_sources instead. Note that configuration-dependent sources are not supported yet (see https://gitlab.kitware.com/cmake/cmake/issues/18233)
target_sources(Build
    PRIVATE
      $<$<PLATFORM_ID:Windows>:source/windows-only.cpp> # Use of generator expressions https://cmake.org/cmake/help/latest/manual/cmake-generator-expressions.7.html
)
if(${PROJECT_NAME}_USE_ADDITIONAL_SOURCEFILE)
    target_sources(Build PRIVATE source/additional-sourcefile.cpp)
    # You can add defines if needed. PRIVATE means it will only be defined for this target. (see transitive usage)
    target_compile_definitions(Build PRIVATE HAS_ADDITIONAL_SOURCEFILE)
endif()


# Require c++17, this is better than setting CMAKE_CXX_STANDARD since it won't pollute other targets
# note : cxx_std_* features were added in CMake 3.8.2
target_compile_features(Build PRIVATE cxx_std_17)
target_compile_options(Build PRIVATE $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-mavx2 -mfma>)

# CMake scripts extensions
target_set_warnings(Build ENABLE ALL AS_ERROR ALL DISABLE Annoying) # Helper that can set default warning flags for you
target_enable_lto(Build optimized) #enable lto if available for non-debug configurations
copy_dlls_for_debug(Build "" "") # Copy dependencies next to the executable (DLLs for example)

# Setup our project as the startup project for Visual so that people don't need to do it manually
set_directory_properties(PROPERTIES VS_STARTUP_PROJECT Build)

#=================================#
#  Llama 3.2 language model example  #
#=================================#

add_executable(LlamaExample
    examples/c++/language-model/llama-3.2-1b.cpp
    source/math/matrix.cpp
    source/core/loss.cpp
    source/core/optimizer.cpp
    source/core/regularizer.cpp
    source/core/tokenizer.cpp
    source/core/token-sampler.cpp
    source/core/embedding-layer.cpp
    source/core/attention-layer.cpp
    source/core/transformer-block.cpp
    source/core/neural-network-layer.cpp
    source/core/layer-norm.cpp
    source/core/rms-norm.cpp
    source/core/sin-pos-encode.cpp
    source/core/rot-pos-encode.cpp
    source/core/masking.cpp
    source/models/transformer.cpp
    source/utils/gguf-loader.cpp
    source/utils/llama-loader.cpp
)

target_include_directories(LlamaExample
    PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/include
        ${CMAKE_CURRENT_LIST_DIR}/source
)

target_link_libraries(LlamaExample
    PRIVATE
        fmt::fmt spdlog::spdlog $<$<BOOL:${OpenMP_CXX_FOUND}>:OpenMP::OpenMP_CXX>
)

target_compile_features(LlamaExample PRIVATE cxx_std_17)
target_compile_options(LlamaExample PRIVATE $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-mavx2 -mfma>)

if(ML_USE_CUDA)
    target_sources(LlamaExample PRIVATE
        source/cuda/cuda-matrix.cu
        source/cuda/cuda-kernels.cu
        source/cuda/gpu-transformer.cu
    )
    target_compile_definitions(LlamaExample PRIVATE ML_USE_CUDA)
    target_link_libraries(LlamaExample PRIVATE CUDA::cublas CUDA::cudart)
    set_target_properties(LlamaExample PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

#=================================#
# SmolLM2 language model example  #
#=================================#

add_executable(SmolLM2Example
    examples/c++/language-model/smollm2-135m.cpp
    source/math/matrix.cpp
    source/core/loss.cpp
    source/core/optimizer.cpp
    source/core/regularizer.cpp
    source/core/tokenizer.cpp
    source/core/token-sampler.cpp
    source/core/embedding-layer.cpp
    source/core/attention-layer.cpp
    source/core/transformer-block.cpp
    source/core/neural-network-layer.cpp
    source/core/layer-norm.cpp
    source/core/rms-norm.cpp
    source/core/sin-pos-encode.cpp
    source/core/rot-pos-encode.cpp
    source/core/masking.cpp
    source/models/transformer.cpp
    source/utils/gguf-loader.cpp
    source/utils/llama-loader.cpp
)

target_include_directories(SmolLM2Example
    PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/include
        ${CMAKE_CURRENT_LIST_DIR}/source
)

target_link_libraries(SmolLM2Example
    PRIVATE
        fmt::fmt spdlog::spdlog $<$<BOOL:${OpenMP_CXX_FOUND}>:OpenMP::OpenMP_CXX>
)

target_compile_features(SmolLM2Example PRIVATE cxx_std_17)
target_compile_options(SmolLM2Example PRIVATE $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-mavx2 -mfma>)

if(ML_USE_CUDA)
    target_sources(SmolLM2Example PRIVATE
        source/cuda/cuda-matrix.cu
        source/cuda/cuda-kernels.cu
        source/cuda/gpu-transformer.cu
    )
    target_compile_definitions(SmolLM2Example PRIVATE ML_USE_CUDA)
    target_link_libraries(SmolLM2Example PRIVATE CUDA::cublas CUDA::cudart)
    set_target_properties(SmolLM2Example PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()


#=================================#
#  SIMD matmul micro-benchmark    #
#=================================#

add_executable(SimdMatmulBenchmark
    examples/c++/benchmarks/simd-matmul-benchmark.cpp
    source/math/matrix.cpp
    include/ml_lib/math/matrix.h
)

target_include_directories(SimdMatmulBenchmark
    PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/include
        ${CMAKE_CURRENT_LIST_DIR}/source
)

target_link_libraries(SimdMatmulBenchmark
    PRIVATE
        $<$<BOOL:${OpenMP_CXX_FOUND}>:OpenMP::OpenMP_CXX>
)

target_compile_features(SimdMatmulBenchmark PRIVATE cxx_std_17)
target_compile_options(SimdMatmulBenchmark PRIVATE $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-O3 -mavx2 -mfma>)

#===============#
#  Foo library  #
#===============#

# Usually libraries are listed before executables, but in this case we only use it for the tests
# Note that we use both the include and source folders for headers, based on wether they are part of the public API or not.
# For more information about this rationale, see https://github.com/vector-of-bool/pitchfork
add_library(bp_foo
    source/foo.cpp
    source/foo-impl.h
    include/foo.h
)

# Since we put the public interface headers in the include directory, we need to tell the compiler so that we can #include <file>.
target_include_directories(bp_foo
    PUBLIC # The folder must be used in the include path for any target using this library
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include> # Due to the way installation work, we only want this path set when building, not once installed
)

# The following properties are useful when you want to have an 'install' target
set_target_properties(${PROJECT_NAME} 
    PROPERTIES 
        PUBLIC_HEADER ${CMAKE_CURRENT_LIST_DIR}/include/foo.h # Headers listed here will automatically be copied when installing. Note that directories hierarchy is not preserved.
        DEBUG_POSTFIX d # We had a postfix so that we can install debug and release libraries side by side (Windows way)
)

# We tell CMake what are the target dependencies
target_link_libraries(bp_foo
    PRIVATE # fmt is only used in implementation, not exposed in public headers
        fmt::fmt # Use the namespaced version to make sure we have the target and not the static lib only (which doesn't have transitive properties)
)
# Give a 'namespaced' name to libraries targets, as it can't be mistaken with system libraries
# Use the same namespace as the one from the install command. This is only really needed if you want to support usage of your library through add_subdirectory.
add_library(${PROJECT_NAME}::foo ALIAS bp_foo)

#===========#
#   Tests   #
#===========#

if(${PROJECT_NAME}_BUILD_TESTS)
    # Let the user add options to the test runner if needed
    set(TEST_RUNNER_PARAMS "--force-colors=true" CACHE STRING "Options to add to our test runners commands")
    # In a real project you most likely want to exclude test folders
    # list(APPEND CUSTOM_COVERAGE_EXCLUDE "/test/")
    add_subdirectory(tests)
    # You can setup some custom variables and add them to the CTestCustom.cmake.in template to have custom ctest settings
    # For example, you can exclude some directories from the coverage reports such as third-parties and tests
    configure_file(
        ${CMAKE_CURRENT_LIST_DIR}/cmake/CTestCustom.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/CTestCustom.cmake
        @ONLY
    )
endif()

#############
## Doxygen ##
#############

if(${PROJECT_NAME}_USE_DOXYGEN AND CMAKE_VERSION VERSION_GREATER_EQUAL 3.9)
    find_package(Doxygen
        OPTIONAL_COMPONENTS dot mscgen dia
    )
    if(DOXYGEN_FOUND)
        set(DOXYGEN_USE_MDFILE_AS_MAINPAGE README.md)
        doxygen_add_docs(
            doc
            README.md source
            COMMENT "Generate man pages"
        )
    endif()
endif()


###############
## Packaging ##
###############

if(${PROJECT_NAME}_INSTALL)
    # If we want to use CPack, we need to include it so that it populates variables from our CMakeLists.txt.
    # This will also create a `package` target on supported build systems (make, ninja, VS).
    # There are various CPACK_* variables you can set before `include(CPack)` to configure it (see https://cmake.org/cmake/help/latest/module/CPack.html#variables-common-to-all-cpack-generators).
    set(CPACK_RESOURCE_FILE_README ${CMAKE_CURRENT_LIST_DIR}/README.md)
    include(CPack)
    
    # Let users choose where to install the cmake package descriptions
    # For that we make use of the CMake Cache
    set(${PROJECT_NAME}_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}" CACHE STRING "Path to install ${PROJECT_NAME} Config*.cmake files to.")
    set(${PROJECT_NAME}_MODULE_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/cmake"               CACHE STRING "Path to install ${PROJECT_NAME}'s .cmake  module files to.")

    # Use version checking helper provided by CMake so that users can safely use a version number in their find_package calls
    write_basic_package_version_file(
        ${PROJECT_NAME}ConfigVersion.cmake # The name of the version file needed by find_package.
        VERSION ${PROJECT_VERSION}         # The version of the project, already set by the `project` command at the top of this file
        COMPATIBILITY SameMajorVersion     # We use semantic versioning, backward compatibity is only guaranteed for a same major version
    )


    # We will need our own file if we have our own dependencies or want some special behavior when the user calls find_package
    # otherwise we could simply install the exports as the ${PROJECT_NAME}Config.cmake
    configure_package_config_file(
        ${PROJECT_SOURCE_DIR}/cmake/${PROJECT_NAME}Config.cmake.in # This is your template file
        ${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake          # This is the resulting file
        INSTALL_DESTINATION ${${PROJECT_NAME}_INSTALL_CMAKEDIR}    # This is where the file will be installed
        # List of paths that needs to be relocated once installed
        # For example if the variable containing the path is named MY_PATH, all instances of @PACKAGE_MY_PATH@ in the template will be replaced by the relocated version of the path
        # This is mostly only needed when you want to install cmake modules or have an unusual layout that cmake is not aware of.
        PATH_VARS ${PROJECT_NAME}_MODULE_INSTALL_DIR # This will be exposed as @PACKAGE_BoilerPlate_MODULE_INSTALL_DIR@ in the template file   
        # Imported targets do not require the following macros
        NO_SET_AND_CHECK_MACRO
        NO_CHECK_REQUIRED_COMPONENTS_MACRO 
    )


    # The following will export the targets under the name ${PROJECT_NAME}_Targets, not install them yet
    # It will then need a call to `install(EXPORT)`
    # Note: bp_foo removed from install because it depends on external fmt library not in export set
    install(
        TARGETS
          Build # We can install executables
        EXPORT ${PROJECT_NAME}_Targets
    # Following is only needed pre-cmake3.14
    #    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    #    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    #    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
		# Note PUBLIC_HEADER DESTINATION is necessary only if you provide RUNTIME/LIBRARY/ARCHIVE otherwise CMake will map it to the INCLUDES destination (CMake will warn you with "INSTALL TARGETS - target BoilerPlate has PUBLIC_HEADER files but no PUBLIC_HEADER DESTINATION." )
	#	PUBLIC_HEADER DESTINATION  ${CMAKE_INSTALL_INCLUDEDIR} 
    # If you want to split between runtime and dev for examples, take a look at COMPONENT, NAMELINK_COMPONENT etc
    # More info in Craig Scott's talk "Deep CMake for library authors" https://www.youtube.com/watch?v=m0DwB4OvDXk
        INCLUDES
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )

    # This time, install all the exported targets under the ${PROJECT_NAME}_Targets name.
    install(
        EXPORT ${PROJECT_NAME}_Targets 
        NAMESPACE ${PROJECT_NAME}::        # Always specify a namespace so that users can make sure they link targets with transitive properties and not only the library
        FILE ${PROJECT_NAME}Targets.cmake  # This is the file that needs to be included from your *Config.cmake. Otherwise, you could just make this your actual *Config.cmake file.
        DESTINATION ${${PROJECT_NAME}_INSTALL_CMAKEDIR}
    )


    # So far we only installed the exported targets, now install the package config files.
    # If you do not list headers in the PUBLIC_HEADER property, you will need to copy them using `install(FILES)` or `install(DIRECTORY)` too.
    # In that case, you can use CMAKE_INSTALL_INCLUDEDIR as the base destination path.
    install(FILES 
        ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake 
        DESTINATION
            ${${PROJECT_NAME}_INSTALL_CMAKEDIR}
    )

endif()

