
# -------------------------------------------------------------------------------------
# add executable
add_executable( DL_Ch02_check_matplotplusplus Ch02_Linear_algebra/Check_matplotplusplus.cpp)

# link libraries
target_link_libraries( DL_Ch02_check_matplotplusplus ${TORCH_LIBRARIES} matplot)
set_target_properties( DL_Ch02_check_matplotplusplus  PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------
add_executable(DL_Ch02_Linear_algebra Ch02_Linear_algebra/Linear_algebra.cpp)
				
target_link_libraries(DL_Ch02_Linear_algebra ${TORCH_LIBRARIES} matplot)
set_target_properties(DL_Ch02_Linear_algebra PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------
add_executable(DL_Ch03_Probability
		Ch03_Probability_and_information_theory/Probability.cpp
		../Utils/helpfunction.h
		../Utils/helpfunction.cpp)
				
target_link_libraries(DL_Ch03_Probability ${TORCH_LIBRARIES} matplot)
set_target_properties(DL_Ch03_Probability PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------
add_executable(DL_Ch03_Information_theory 
		Ch03_Probability_and_information_theory/Information_theory.cpp
		../Utils/helpfunction.h
		../Utils/helpfunction.cpp)
				
target_link_libraries(DL_Ch03_Information_theory ${TORCH_LIBRARIES} matplot)
set_target_properties(DL_Ch03_Information_theory PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------
add_executable(DL_Ch04_Numerical_computation 
		Ch04_Numerical_computation/Numerical_computation.cpp
		../Utils/helpfunction.h
		../Utils/helpfunction.cpp)
				
target_link_libraries(DL_Ch04_Numerical_computation ${TORCH_LIBRARIES} matplot)
set_target_properties(DL_Ch04_Numerical_computation PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------
add_executable(DL_Ch05_Machine_learning_basics 
		Ch05_Machine_learning_basics/Machine_learning_basics.cpp
		../Utils/helpfunction.h
		../Utils/helpfunction.cpp)
				
target_link_libraries(DL_Ch05_Machine_learning_basics ${TORCH_LIBRARIES} matplot)
set_target_properties(DL_Ch05_Machine_learning_basics PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------
add_executable(DL_Ch05_Supervised_learning_algorithms 
		Ch05_Machine_learning_basics/Supervised_learning_algorithms.cpp
		../Utils/helpfunction.h
		../Utils/helpfunction.cpp)
				
target_link_libraries(DL_Ch05_Supervised_learning_algorithms ${TORCH_LIBRARIES} matplot)
set_target_properties(DL_Ch05_Supervised_learning_algorithms PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------
add_executable(DL_Ch05_Unsupervised_learning_algorithms 
		Ch05_Machine_learning_basics/Unsupervised_learning_algorithms.cpp
		../Utils/helpfunction.h
		../Utils/helpfunction.cpp)
				
target_link_libraries(DL_Ch05_Unsupervised_learning_algorithms ${TORCH_LIBRARIES} matplot)
set_target_properties(DL_Ch05_Unsupervised_learning_algorithms PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------
add_executable(DL_Ch06_MLP 
		Ch06_Deep_feedforward_networks/MLP.cpp
		../Utils/helpfunction.h
		../Utils/helpfunction.cpp
		../Utils/fashion.cpp
		../Utils/fashion.h)
				
target_link_libraries(DL_Ch06_MLP ${TORCH_LIBRARIES} matplot)
set_target_properties(DL_Ch06_MLP PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------
add_executable(DL_Ch06_XOR 
		Ch06_Deep_feedforward_networks/XOR.cpp
		../Utils/helpfunction.h
		../Utils/helpfunction.cpp)
				
target_link_libraries(DL_Ch06_XOR ${TORCH_LIBRARIES} matplot)
set_target_properties(DL_Ch06_XOR PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------
add_executable(DL_Ch06_activation_demo 
		Ch06_Deep_feedforward_networks/activation_function_demo.cpp
		../Utils/helpfunction.h
		../Utils/helpfunction.cpp)
				
target_link_libraries(DL_Ch06_activation_demo ${TORCH_LIBRARIES} matplot)
set_target_properties(DL_Ch06_activation_demo PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------
add_executable(DL_Ch07_Regularization
		Ch07_Regularization/Regularization.cpp
		../Utils/helpfunction.h
		../Utils/helpfunction.cpp)
				
target_link_libraries(DL_Ch07_Regularization ${TORCH_LIBRARIES} matplot)
set_target_properties(DL_Ch07_Regularization PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------
add_executable(DL_Ch07_dropout
		Ch07_Regularization/dropout.cpp
		../Utils/helpfunction.h
		../Utils/helpfunction.cpp
		../Utils/fashion.cpp
		../Utils/fashion.h)
				
target_link_libraries(DL_Ch07_dropout ${TORCH_LIBRARIES} matplot)
set_target_properties(DL_Ch07_dropout PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------
add_executable(DL_Ch07_weight_decay
		Ch07_Regularization/weight_decay.cpp
		../Utils/helpfunction.h
		../Utils/helpfunction.cpp)
				
target_link_libraries(DL_Ch07_weight_decay ${TORCH_LIBRARIES} matplot)
set_target_properties(DL_Ch07_weight_decay PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------
add_executable(DL_Ch07_dataset_augmentation 
		Ch07_Regularization/Dataset_augmentation.cpp
		../Utils/helpfunction.h
		../Utils/helpfunction.cpp
		../Utils/fashion.cpp
		../Utils/fashion.h)
				
target_link_libraries(DL_Ch07_dataset_augmentation ${OpenCV_LIBS} ${TORCH_LIBRARIES} matplot)
set_target_properties(DL_Ch07_dataset_augmentation PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------
add_executable(DL_Ch07_RandomForest
		Ch07_Regularization/RandomForest.cpp
		../Utils/helpfunction.h
		../Utils/helpfunction.cpp)
				
target_link_libraries(DL_Ch07_RandomForest ${TORCH_LIBRARIES} matplot)
set_target_properties(DL_Ch07_RandomForest PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------
add_executable(DL_Ch07_Bagging
		Ch07_Regularization/Bagging.cpp
		../Utils/helpfunction.h
		../Utils/helpfunction.cpp)
				
target_link_libraries(DL_Ch07_Bagging ${TORCH_LIBRARIES} matplot)
set_target_properties(DL_Ch07_Bagging PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------
add_executable(DL_Ch08_Optimization_demo
		Ch08_Optimization/Optimization_demo.cpp
		../Utils/helpfunction.h
		../Utils/helpfunction.cpp
		../Utils/fashion.cpp
		../Utils/fashion.h)
				
target_link_libraries(DL_Ch08_Optimization_demo ${TORCH_LIBRARIES} matplot)
set_target_properties(DL_Ch08_Optimization_demo PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------
add_executable(DL_Ch08_BatchNormalization
		Ch08_Optimization/BatchNormalization.cpp
		../Utils/helpfunction.h
		../Utils/helpfunction.cpp)
				
target_link_libraries(DL_Ch08_BatchNormalization ${TORCH_LIBRARIES} matplot)
set_target_properties(DL_Ch08_BatchNormalization PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------
add_executable(DL_Ch08_GradientDescent
		Ch08_Optimization/GradientDescent.cpp
		../Utils/helpfunction.h
		../Utils/helpfunction.cpp)
				
target_link_libraries(DL_Ch08_GradientDescent ${TORCH_LIBRARIES} matplot)
set_target_properties(DL_Ch08_GradientDescent PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------
add_executable(DL_Ch09_Convolutional_networks
		Ch09_Convolutional_networks/ConvolutionalNetworks.cpp
		../Utils/helpfunction.h
		../Utils/helpfunction.cpp
		../Utils/fashion.cpp
		../Utils/fashion.h)
				
target_link_libraries(DL_Ch09_Convolutional_networks ${TORCH_LIBRARIES} matplot)
set_target_properties(DL_Ch09_Convolutional_networks PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------
add_executable(DL_Ch10_RecurrentNeuralNetworks
		Ch10_SequenceModeling/RecurrentNeuralNetworks.cpp
		../Utils/helpfunction.h
		../Utils/helpfunction.cpp
		../Utils/fashion.cpp
		../Utils/fashion.h)
				
target_link_libraries(DL_Ch10_RecurrentNeuralNetworks ${TORCH_LIBRARIES} matplot)
set_target_properties(DL_Ch10_RecurrentNeuralNetworks PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------
add_executable(DL_Ch10_RNN
		Ch10_SequenceModeling/RNN.cpp
		../Utils/helpfunction.h
		../Utils/helpfunction.cpp)
				
target_link_libraries(DL_Ch10_RNN ${TORCH_LIBRARIES} matplot)
set_target_properties(DL_Ch10_RNN PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------
add_executable(DL_Ch10_GRU
		Ch10_SequenceModeling/GatedRecurrentUnits.cpp
		../Utils/helpfunction.h
		../Utils/helpfunction.cpp)
				
target_link_libraries(DL_Ch10_GRU ${TORCH_LIBRARIES} matplot)
set_target_properties(DL_Ch10_GRU PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------
add_executable(DL_Ch10_LSTM
		Ch10_SequenceModeling/LongShortTermMemory.cpp
		../Utils/helpfunction.h
		../Utils/helpfunction.cpp)
				
target_link_libraries(DL_Ch10_LSTM ${TORCH_LIBRARIES} matplot)
set_target_properties(DL_Ch10_LSTM PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------
add_executable(DL_Ch10_1D-Convolutions
		Ch10_SequenceModeling/1D_Convolutions.cpp
		../Utils/helpfunction.h
		../Utils/helpfunction.cpp)
				
target_link_libraries(DL_Ch10_1D-Convolutions ${TORCH_LIBRARIES} matplot)
set_target_properties(DL_Ch10_1D-Convolutions PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------
add_executable(DL_Ch10_VariableLengthSequences
		Ch10_SequenceModeling/VariableLengthSequences.cpp
		../Utils/helpfunction.h
		../Utils/helpfunction.cpp)
				
target_link_libraries(DL_Ch10_VariableLengthSequences ${TORCH_LIBRARIES} matplot)
set_target_properties(DL_Ch10_VariableLengthSequences PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------
add_executable(DL_Ch10_Sequence2Sequence
		Ch10_SequenceModeling/Sequence2Sequence.cpp
		../Utils/helpfunction.h
		../Utils/helpfunction.cpp)
				
target_link_libraries(DL_Ch10_Sequence2Sequence ${TORCH_LIBRARIES} matplot)
set_target_properties(DL_Ch10_Sequence2Sequence PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------
add_executable(DL_Ch10_Attention
		Ch10_SequenceModeling/Attention.cpp
		../Utils/helpfunction.h
		../Utils/helpfunction.cpp)
				
target_link_libraries(DL_Ch10_Attention ${TORCH_LIBRARIES} matplot)
set_target_properties(DL_Ch10_Attention PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------
add_executable(DL_Ch11_PracticalMethodology
		Ch11_PracticalMethodology/PracticalMethodology.cpp
		../Utils/helpfunction.h
		../Utils/helpfunction.cpp)
				
target_link_libraries(DL_Ch11_PracticalMethodology ${TORCH_LIBRARIES} matplot)
set_target_properties(DL_Ch11_PracticalMethodology PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------
add_executable(DL_Ch11_HyperparameterOptimization
		Ch11_PracticalMethodology/HyperparameterOptimization.cpp
		../Utils/helpfunction.h
		../Utils/helpfunction.cpp)
				
target_link_libraries(DL_Ch11_HyperparameterOptimization ${OpenCV_LIBS} ${TORCH_LIBRARIES} matplot)
set_target_properties(DL_Ch11_HyperparameterOptimization PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------
add_executable(DL_Ch12_ClassSpecificImageGeneration
		Ch12_Applications/ClassSpecificImageGeneration.cpp
		../Utils/helpfunction.h
		../Utils/helpfunction.cpp
		../Utils/opencv_helpfunctions.h
		../Utils/opencv_helpfunctions.cpp)
				
target_link_libraries(DL_Ch12_ClassSpecificImageGeneration ${OpenCV_LIBS} ${TORCH_LIBRARIES} matplot)
set_target_properties(DL_Ch12_ClassSpecificImageGeneration PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------
add_executable(DL_Ch12_NeuralMachineTranslation
		Ch12_Applications/NeuralMachineTranslation.cpp
		../Utils/helpfunction.h
		../Utils/helpfunction.cpp)
				
target_link_libraries(DL_Ch12_NeuralMachineTranslation ${TORCH_LIBRARIES} matplot)
set_target_properties(DL_Ch12_NeuralMachineTranslation PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------
add_executable(DL_Ch13_Probabilstic_PCA
		Ch13_Linear_factor_models/Probabilstic_PCA.cpp
		../Utils/helpfunction.h
		../Utils/helpfunction.cpp)
				
target_link_libraries(DL_Ch13_Probabilstic_PCA ${TORCH_LIBRARIES} matplot)
set_target_properties(DL_Ch13_Probabilstic_PCA PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------
add_executable(DL_Ch14_RegularizedAutoEncoder
		Ch14_Autoencoders/RegularizedAutoEncoder.cpp
		../Utils/helpfunction.h
		../Utils/helpfunction.cpp
		../Utils/fashion.h
		../Utils/fashion.cpp)
				
target_link_libraries(DL_Ch14_RegularizedAutoEncoder ${TORCH_LIBRARIES} matplot)
set_target_properties(DL_Ch14_RegularizedAutoEncoder PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------
add_executable(DL_Ch14_DenoisingAutoencoder
		Ch14_Autoencoders/DenoisingAutoencoder.cpp
		../Utils/helpfunction.h
		../Utils/helpfunction.cpp
		../Utils/fashion.h
		../Utils/fashion.cpp)
				
target_link_libraries(DL_Ch14_DenoisingAutoencoder ${TORCH_LIBRARIES} matplot)
set_target_properties(DL_Ch14_DenoisingAutoencoder PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------























