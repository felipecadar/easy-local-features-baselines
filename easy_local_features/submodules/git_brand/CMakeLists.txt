cmake_minimum_required (VERSION 2.6)
project (BRAND_DESCRIPTOR_DEMO)

find_package (PCL 1.4 REQUIRED)
find_package (OpenCV REQUIRED) 

#set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
#set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_BUILD_TYPE Release)

include_directories (${PCL_INCLUDE_DIRS})
 
add_executable ( brand_match_demo brand_match_demo.cpp brand.cpp )
target_link_libraries ( brand_match_demo
 #  pcl_features
 #  pcl_common
 #  pcl_search
 #  pcl_kdtree
   ${PCL_LIBRARIES}
   ${OpenCV_LIBS} )

add_executable ( desc_brand desc_brand.cpp brand.cpp )
target_link_libraries ( desc_brand
 #  pcl_features
 #  pcl_common
 #  pcl_search
 #  pcl_kdtree
   ${PCL_LIBRARIES}
   ${OpenCV_LIBS} )
