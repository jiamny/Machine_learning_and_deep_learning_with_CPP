
# -------------------------------------------------------------------------------------
# add executable
add_executable( UDL_Ch01_BackgroundMaths 
	Ch01_BackgroundMaths/BackgroundMaths.cpp
	../Utils/helpfunction.h
	../Utils/helpfunction.cpp
)

# link libraries
target_link_libraries( UDL_Ch01_BackgroundMaths ${TORCH_LIBRARIES} matplot)
set_target_properties( UDL_Ch01_BackgroundMaths  PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------
# add executable
add_executable( UDL_Ch02_SupervisedLearning 
	Ch02_SupervisedLearning/SupervisedLearning.cpp
	../Utils/helpfunction.h
	../Utils/helpfunction.cpp
)

# link libraries
target_link_libraries( UDL_Ch02_SupervisedLearning ${TORCH_LIBRARIES} matplot)
set_target_properties( UDL_Ch02_SupervisedLearning  PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------
# add executable
add_executable( UDL_Ch03_ShallowNetworks_I 
	Ch03_ShallowNeuralNetworks/ShallowNetworks_I.cpp
	../Utils/helpfunction.h
	../Utils/helpfunction.cpp
)

# link libraries
target_link_libraries( UDL_Ch03_ShallowNetworks_I ${TORCH_LIBRARIES} matplot)
set_target_properties( UDL_Ch03_ShallowNetworks_I  PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------
# add executable
add_executable( UDL_Ch03_ShallowNetworks_II 
	Ch03_ShallowNeuralNetworks/ShallowNetworks_II.cpp
	../Utils/helpfunction.h
	../Utils/helpfunction.cpp
)

# link libraries
target_link_libraries( UDL_Ch03_ShallowNetworks_II ${TORCH_LIBRARIES} matplot)
set_target_properties( UDL_Ch03_ShallowNetworks_II  PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------






















