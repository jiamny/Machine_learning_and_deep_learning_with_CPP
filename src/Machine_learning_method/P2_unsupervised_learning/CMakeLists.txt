

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

add_executable(ML_P2_unsupervised_learning_ch18_PLSA Ch18_PLSA/PLSA.cpp
	../../Utils/helpfunction.h
	../../Utils/helpfunction.cpp)
				
target_link_libraries(ML_P2_unsupervised_learning_ch18_PLSA ${TORCH_LIBRARIES} matplot)
set_target_properties(ML_P2_unsupervised_learning_ch18_PLSA PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------

add_executable(ML_P2_unsupervised_learning_ch19_Gibbs_sampling Ch19_MCMC_MonteCarlo/Gibbs_sampling.cpp
	../../Utils/helpfunction.h
	../../Utils/helpfunction.cpp)
				
target_link_libraries(ML_P2_unsupervised_learning_ch19_Gibbs_sampling ${TORCH_LIBRARIES} matplot)
set_target_properties(ML_P2_unsupervised_learning_ch19_Gibbs_sampling PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------

add_executable(ML_P2_unsupervised_learning_ch19_MetropolisHastings Ch19_MCMC_MonteCarlo/MetropolisHastings.cpp
	../../Utils/helpfunction.h
	../../Utils/helpfunction.cpp)
				
target_link_libraries(ML_P2_unsupervised_learning_ch19_MetropolisHastings ${TORCH_LIBRARIES} matplot)
set_target_properties(ML_P2_unsupervised_learning_ch19_MetropolisHastings PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------

add_executable(ML_P2_unsupervised_learning_ch19_MonteCarloIntegration Ch19_MCMC_MonteCarlo/MonteCarloIntegration.cpp
	../../Utils/helpfunction.h
	../../Utils/helpfunction.cpp)
				
target_link_libraries(ML_P2_unsupervised_learning_ch19_MonteCarloIntegration ${TORCH_LIBRARIES} matplot)
set_target_properties(ML_P2_unsupervised_learning_ch19_MonteCarloIntegration PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------

add_executable(ML_P2_unsupervised_learning_ch19_MCMC Ch19_MCMC_MonteCarlo/MCMC.cpp
	../../Utils/helpfunction.h
	../../Utils/helpfunction.cpp)
				
target_link_libraries(ML_P2_unsupervised_learning_ch19_MCMC ${TORCH_LIBRARIES} matplot)
set_target_properties(ML_P2_unsupervised_learning_ch19_MCMC PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------

add_executable(ML_P2_unsupervised_learning_ch20_GibbsSamplingLDA Ch20_LDA/GibbsSamplingLDA.cpp
	../../Utils/helpfunction.h
	../../Utils/helpfunction.cpp)
				
target_link_libraries(ML_P2_unsupervised_learning_ch20_GibbsSamplingLDA ${TORCH_LIBRARIES} matplot)
set_target_properties(ML_P2_unsupervised_learning_ch20_GibbsSamplingLDA PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------

add_executable(ML_P2_unsupervised_learning_ch20_LDA Ch20_LDA/LDA.cpp
	../../Utils/helpfunction.h
	../../Utils/helpfunction.cpp)
				
target_link_libraries(ML_P2_unsupervised_learning_ch20_LDA ${TORCH_LIBRARIES} matplot)
set_target_properties(ML_P2_unsupervised_learning_ch20_LDA PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------

add_executable(ML_P2_unsupervised_learning_ch21_PageRank Ch21_PageRank/PageRank.cpp
	../../Utils/helpfunction.h
	../../Utils/helpfunction.cpp)
				
target_link_libraries(ML_P2_unsupervised_learning_ch21_PageRank ${TORCH_LIBRARIES} matplot)
set_target_properties(ML_P2_unsupervised_learning_ch21_PageRank PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------












