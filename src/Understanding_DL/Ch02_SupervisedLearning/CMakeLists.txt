
# -------------------------------------------------------------------------------------
# add executable
add_executable( UDL_Ch02_SupervisedLearning 
	SupervisedLearning.cpp
	../../Utils/helpfunction.h
	../../Utils/helpfunction.cpp
)

# link libraries
target_link_libraries( UDL_Ch02_SupervisedLearning ${TORCH_LIBRARIES} matplot)
set_target_properties( UDL_Ch02_SupervisedLearning  PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------









