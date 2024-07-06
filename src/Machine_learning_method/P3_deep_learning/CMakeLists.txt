

# -------------------------------------------------------------------------------------
add_executable(ML_P3_deep_learning_Ch23_FFN Ch23_Feedforward_networks/feedforward_nn.cpp
	../../Utils/helpfunction.h
	../../Utils/helpfunction.cpp)
				
target_link_libraries(ML_P3_deep_learning_Ch23_FFN ${TORCH_LIBRARIES} matplot)
set_target_properties(ML_P3_deep_learning_Ch23_FFN PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------
add_executable(ML_P3_deep_learning_Ch23_NeuralNetMLP Ch23_Feedforward_networks/NeuralNetMLP.cpp
	../../Utils/helpfunction.h
	../../Utils/helpfunction.cpp
	../../Utils/fashion.cpp
	../../Utils/fashion.h)
				
target_link_libraries(ML_P3_deep_learning_Ch23_NeuralNetMLP ${TORCH_LIBRARIES} matplot)
set_target_properties(ML_P3_deep_learning_Ch23_NeuralNetMLP PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------
add_executable(ML_P3_deep_learning_Ch24_CNN_classifying Ch24_CNN/CNN_classifying.cpp
	../../Utils/helpfunction.h
	../../Utils/helpfunction.cpp
	../../Utils/opencv_helpfunctions.h
	../../Utils/opencv_helpfunctions.cpp
	../../Utils/fashion.cpp
	../../Utils/fashion.h)
				
target_link_libraries(ML_P3_deep_learning_Ch24_CNN_classifying ${OpenCV_LIBS} ${TORCH_LIBRARIES} matplot)
set_target_properties(ML_P3_deep_learning_Ch24_CNN_classifying PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------
#add_executable(ML_P3_deep_learning_Ch24_tokenizer_example Ch24_CNN/tokenizer_example.cpp
#	../../Utils/helpfunction.h
#	../../Utils/helpfunction.cpp
#	../../tokenizers/tokenizer.cpp 
#	../../tokenizers/unilib/unicode.cpp 
#	../../tokenizers/unilib//uninorms.cpp 
#	)

#if(Boost_FOUND)
#    target_link_libraries(ML_P3_deep_learning_Ch24_tokenizer_example ${Boost_LIBRARIES})
#endif()

#set(unicode_dir "../../tokenizers/unilib") 

#target_include_directories(ML_P3_deep_learning_Ch24_tokenizer_example PUBLIC ${unicode_dir})
		
#target_link_libraries(ML_P3_deep_learning_Ch24_tokenizer_example ${TORCH_LIBRARIES} matplot)
#set_target_properties(ML_P3_deep_learning_Ch24_tokenizer_example PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------
add_executable(ML_P3_deep_learning_Ch25_Sequences Ch25_RNN/Sequences.cpp
	../../Utils/helpfunction.h
	../../Utils/helpfunction.cpp
	../../Utils/opencv_helpfunctions.h
	../../Utils/opencv_helpfunctions.cpp)
				
target_link_libraries(ML_P3_deep_learning_Ch25_Sequences ${OpenCV_LIBS} ${TORCH_LIBRARIES} matplot)
set_target_properties(ML_P3_deep_learning_Ch25_Sequences PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------
add_executable(ML_P3_deep_learning_Ch25_LSTMTagger Ch25_RNN/LSTMTagger.cpp
	../../Utils/helpfunction.h
	../../Utils/helpfunction.cpp
	../../Utils/opencv_helpfunctions.h
	../../Utils/opencv_helpfunctions.cpp)
				
target_link_libraries(ML_P3_deep_learning_Ch25_LSTMTagger ${OpenCV_LIBS} ${TORCH_LIBRARIES} matplot)
set_target_properties(ML_P3_deep_learning_Ch25_LSTMTagger PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)



