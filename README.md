# MNIST-CNN

This repository contains a Jupyter Notebook and python file. Both of these contain all the code to split the data into training, validation, testing sets; to outline, optimise, and train the model; and test the final version of the CNN model. The notebook also contains all the outputs and results from the final iteration of the model whilst the python file only contains the code. 

The data being used is the MNIST dataset provided through TensorFlow. This dataset is often described as the ‘Hello World’ of image classification tasks so it seems like a suitable place to start.

The model consists of 2 2D Convolution layers, the first performing 32 convolutions and the other 2 performing 64., as well as 2 2DPooling layers, the first being an AveragePool and the second being a MaxPool. These are followed by a Flattening layer and then a 128 node Dense layer. The GeLu activation function is used in all the Convolution layers and the Dense layer. This function was chosen after multiple iterations of testing and was found to produce the lowest loss. Finally there is a 10 node Dense layer which uses the SoftMax activation function as this is a classification task.

For compilation, the model uses the Adam optimisation function and the Sparse Categorical Cross-entropy loss function.

Model training uses 2 stopping hyperparameters where training stops whenever the first of these is reached. The hyperparameters are Num Epochs which is set to 20, and the Early Stopping callback function with a patience of 2. 
 
The final model required 14 Epochs to train and produced a validation loss of 0.0058 and a validation accuracy of 99.78%. It was these values that were used to measure the impacts of changes to the model. Even though this runs the risk of overfitting the model to the training and validation data, it is hoped that combining 2 stopping parameters would reduce this risk. 

On the unseen testing dataset the model produced a loss of 0.0328 and an accuracy of 99.24%. These results are worse than the results from the validation data but that is to be expected. 
