

# -------------------------------------------------------------------------------------

add_executable(ML_P4_other_Linear_regression Linear_regression/Linear_regression.cpp
	../../Utils/helpfunction.h
	../../Utils/helpfunction.cpp)
				
target_link_libraries(ML_P4_other_Linear_regression ${TORCH_LIBRARIES} matplot)
set_target_properties(ML_P4_other_Linear_regression PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------

add_executable(ML_P4_other_Random_forest Random_forest/RandomForest.cpp
	../../Utils/helpfunction.h
	../../Utils/helpfunction.cpp)
				
target_link_libraries(ML_P4_other_Random_forest ${TORCH_LIBRARIES} matplot)
set_target_properties(ML_P4_other_Random_forest PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------

add_executable(ML_P4_other_TSNE TSNE/TSNE.cpp
	../../Utils/helpfunction.h
	../../Utils/helpfunction.cpp)
				
target_link_libraries(ML_P4_other_TSNE ${TORCH_LIBRARIES} matplot)
set_target_properties(ML_P4_other_TSNE PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------

add_executable(ML_P4_other_DBSCAN DBSCAN/DBSCAN.cpp
	../../Utils/helpfunction.h
	../../Utils/helpfunction.cpp)
				
target_link_libraries(ML_P4_other_DBSCAN ${TORCH_LIBRARIES} matplot)
set_target_properties(ML_P4_other_DBSCAN PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------


