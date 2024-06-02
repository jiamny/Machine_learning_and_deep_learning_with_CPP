

# -------------------------------------------------------------------------------------

add_executable(ML_P2_unsupervised_learning_ch14_KMeans Ch14_Clustering/KMeans_clustering.cpp
	../../Utils/helpfunction.h
	../../Utils/helpfunction.cpp)
				
target_link_libraries(ML_P2_unsupervised_learning_ch14_KMeans ${TORCH_LIBRARIES} matplot)
set_target_properties(ML_P2_unsupervised_learning_ch14_KMeans PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------

add_executable(ML_P2_unsupervised_learning_ch15_SVD Ch15_SVD/SVD.cpp
	../../Utils/helpfunction.h
	../../Utils/helpfunction.cpp)
				
target_link_libraries(ML_P2_unsupervised_learning_ch15_SVD ${TORCH_LIBRARIES} matplot)
set_target_properties(ML_P2_unsupervised_learning_ch15_SVD PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------

add_executable(ML_P2_unsupervised_learning_ch16_PCA Ch16_PCA/pca.cpp)
				
target_link_libraries(ML_P2_unsupervised_learning_ch16_PCA ${TORCH_LIBRARIES} matplot)
set_target_properties(ML_P2_unsupervised_learning_ch16_PCA PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------

add_executable(ML_P2_unsupervised_learning_ch17_LatentSemanticAnalysis Ch17_LSA/LatentSemanticAnalysis.cpp
	../../Utils/helpfunction.h
	../../Utils/helpfunction.cpp)
				
target_link_libraries(ML_P2_unsupervised_learning_ch17_LatentSemanticAnalysis ${TORCH_LIBRARIES} matplot)
set_target_properties(ML_P2_unsupervised_learning_ch17_LatentSemanticAnalysis PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------

add_executable(ML_P2_unsupervised_learning_ch17_LSA Ch17_LSA/LSA.cpp
	../../Utils/helpfunction.h
	../../Utils/helpfunction.cpp)
				
target_link_libraries(ML_P2_unsupervised_learning_ch17_LSA ${TORCH_LIBRARIES} matplot)
set_target_properties(ML_P2_unsupervised_learning_ch17_LSA PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------

add_executable(ML_P2_unsupervised_learning_ch18_EM_PLSA Ch18_PLSA/EM_based_PLSA.cpp
	../../Utils/helpfunction.h
	../../Utils/helpfunction.cpp)
				
target_link_libraries(ML_P2_unsupervised_learning_ch18_EM_PLSA ${TORCH_LIBRARIES} matplot)
set_target_properties(ML_P2_unsupervised_learning_ch18_EM_PLSA PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------






