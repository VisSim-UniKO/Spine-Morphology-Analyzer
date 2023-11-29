cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(SpineMorphology)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://github.com/VisSim-UniKO/Spine-Morphology-Analyzer")
set(EXTENSION_CATEGORY "VisSim")
set(EXTENSION_CONTRIBUTORS "Vinzent Rittel (Research Group VisSim)")
set(EXTENSION_DESCRIPTION "Tools to measure and alter the human vertebra morphology.")
set(EXTENSION_ICONURL "https://www.example.com/Slicer/Extensions/SpineCurvature.png")
set(EXTENSION_SCREENSHOTURLS "https://www.example.com/Slicer/Extensions/SpineCurvature/Screenshots/1.png")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(Dimensions)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
