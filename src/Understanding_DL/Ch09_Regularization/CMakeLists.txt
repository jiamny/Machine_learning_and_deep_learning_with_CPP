
# -------------------------------------------------------------------------------------
# add executable
add_executable( UDL_Ch09_1_L2_Regularization
	C9_1_L2_Regularization.cpp
	../../Utils/helpfunction.h
	../../Utils/helpfunction.cpp
	../../Utils/UDL_util.h
	../../Utils/UDL_util.cpp
)

# link libraries
target_link_libraries( UDL_Ch09_1_L2_Regularization ${TORCH_LIBRARIES} matplot)
set_target_properties( UDL_Ch09_1_L2_Regularization PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------
# add executable
add_executable( UDL_Ch09_2_Implicit_regularization
	C9_2_Implicit_regularization.cpp
	../../Utils/helpfunction.h
	../../Utils/helpfunction.cpp
	../../Utils/UDL_util.h
	../../Utils/UDL_util.cpp
)

# link libraries
target_link_libraries( UDL_Ch09_2_Implicit_regularization ${TORCH_LIBRARIES} matplot)
set_target_properties( UDL_Ch09_2_Implicit_regularization PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------
# add executable
add_executable( UDL_Ch09_3_Ensembling
	C9_3_Ensembling.cpp
	../../Utils/helpfunction.h
	../../Utils/helpfunction.cpp
	../../Utils/UDL_util.h
	../../Utils/UDL_util.cpp
)

# link libraries
target_link_libraries( UDL_Ch09_3_Ensembling ${TORCH_LIBRARIES} matplot)
set_target_properties( UDL_Ch09_3_Ensembling PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------
# add executable
add_executable( UDL_Ch09_4_Bayesian_approach
	C9_4_Bayesian_approach.cpp
	../../Utils/helpfunction.h
	../../Utils/helpfunction.cpp
	../../Utils/UDL_util.h
	../../Utils/UDL_util.cpp
)

# link libraries
target_link_libraries( UDL_Ch09_4_Bayesian_approach ${TORCH_LIBRARIES} matplot)
set_target_properties( UDL_Ch09_4_Bayesian_approach PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------
# add executable
add_executable( UDL_Ch09_5_Augmentation
	C9_5_Augmentation.cpp
	../../Utils/helpfunction.h
	../../Utils/helpfunction.cpp
)

# link libraries
target_link_libraries( UDL_Ch09_5_Augmentation ${TORCH_LIBRARIES} matplot)
set_target_properties( UDL_Ch09_5_Augmentation PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------







