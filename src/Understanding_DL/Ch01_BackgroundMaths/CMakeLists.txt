
# -------------------------------------------------------------------------------------
# add executable
add_executable( UDL_Ch01_BackgroundMaths 
	BackgroundMaths.cpp
	../../Utils/helpfunction.h
	../../Utils/helpfunction.cpp
)

# link libraries
target_link_libraries( UDL_Ch01_BackgroundMaths ${TORCH_LIBRARIES} matplot)
set_target_properties( UDL_Ch01_BackgroundMaths  PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)











