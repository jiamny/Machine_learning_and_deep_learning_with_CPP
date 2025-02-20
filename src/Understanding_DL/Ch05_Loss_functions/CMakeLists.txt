
# -------------------------------------------------------------------------------------
# add executable
add_executable( UDL_Ch05_1_Least_Squares_Loss
	C5_1_Least_Squares_Loss.cpp
	../../Utils/helpfunction.h
	../../Utils/helpfunction.cpp
)

# link libraries
target_link_libraries( UDL_Ch05_1_Least_Squares_Loss ${TORCH_LIBRARIES} matplot)
set_target_properties( UDL_Ch05_1_Least_Squares_Loss PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------
# add executable
add_executable( UDL_Ch05_2_Binary_Cross_Entropy_Loss
	C5_2_Binary_Cross_Entropy_Loss.cpp
	../../Utils/helpfunction.h
	../../Utils/helpfunction.cpp
)

# link libraries
target_link_libraries( UDL_Ch05_2_Binary_Cross_Entropy_Loss ${TORCH_LIBRARIES} matplot)
set_target_properties( UDL_Ch05_2_Binary_Cross_Entropy_Loss PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------
# add executable
add_executable( UDL_Ch05_3_Multiclass_Cross_entropy_Loss
	C5_3_Multiclass_Cross_entropy_Loss.cpp
	../../Utils/helpfunction.h
	../../Utils/helpfunction.cpp
)

# link libraries
target_link_libraries( UDL_Ch05_3_Multiclass_Cross_entropy_Loss ${TORCH_LIBRARIES} matplot)
set_target_properties( UDL_Ch05_3_Multiclass_Cross_entropy_Loss PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------









