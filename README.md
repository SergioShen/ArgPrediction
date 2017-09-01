# Argument Prediction
A model for argument prediction, using RNN(LSTM) and N-Gram.

## File description
* Python Files:
	* rnn.py: Train and test the LSTM model
	* rnn_ngram.py: Test the performance of RNN-NGRAM combined model
	* input_data.py: Functions for reading data from data set
	* reader.py: Functions for getting data for training and testing
	* cmd.py: Functions for reading N-Gram prediction from Java program
* Data Set:
	* c0810.txt: data set
	* dictionary0810.txt: vocabulary of the data set
* N-Gram Models:
	* train\_set\_k.model: N-Gram model for k-th fold
	* 3GramModel.jar: Runnable jar file. Used to get N-Gram prediction
	
