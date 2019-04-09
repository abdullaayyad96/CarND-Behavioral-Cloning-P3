# **Behavioral Cloning** 

[//]: # (Image References)

[image1]: ./report_figures/data_samples.jpg "Sample Images"
[image2]: ./report_figures/data_before_balancing.jpg "Data Bar Graph"
[image3]: ./report_figures/data_after_balancing.jpg "Balanced Data Bar Graph"


## Project Overciew

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

The rubric points found [here](https://review.udacity.com/#!/rubrics/432/view) were considered throughout this project.


### Model Architecture and Training Strategy

#### 1. Final NN Model Architecture

A deep learning model was implemented in keras for the purpose of this project. The implementation of the deep learning structure can be seen in this [jupyter notebook](/model.ipynb). A description of the Neural Network model I built can be seen below:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         | 160*320*3 image   							| 
| Normalization |   |
| Cropping      |   |
| Convolution   | 1x1 stride, same padding, filter size: 5x5, output channels: 6	|
| Max pooling	  | 2x2 window, 2x2 stride  |
| RELU					|												|
| Convolution   | 1x1 stride, same padding, filter size: 5x5, output channels: 24	|
| Max pooling	 	| 2x2 window, 2x2 stride  |
| RELU					|					
| Convolution  	| 1x1 stride, same padding, filter size: 5x5, output channels: 36	|
| Max pooling	 	| 2x2 window, 2x2 stride  |
| RELU					|					
| Convolution  	| 1x1 stride, same padding, filter size: 5x5, output channels: 48	|
| Max pooling	  | 2x2 window, 2x2 stride  |
| RELU					|					
| Convolution  	| 1x1 stride, same padding, filter size: 3x3, output channels: 64	|
| Max pooling  	| 2x2 window, 2x2 stride  |
| RELU					|		
| Convolution  	| 1x1 stride, same padding, filter size: 3x3, output channels: 64	|
| Max pooling 	| 2x2 window, 2x2 stride  |
| RELU					|				
| Flatten       | Flatten output of 5'th and 6'th Convolutional layers
| Concatenate  | Concatenate flattened output of  5'th and 6'th Convolutional layers
| Fully connected		|  outputs 1064    |
| RELU      |          |
| Dropout   |         |
| Fully connected		|  outputs 100    |
| RELU      |          |
| Dropout   |         |
| Fully connected		|  outputs 50    |
| RELU      |          |
| Dropout   |         |
| Regressor	|  outputs: steering angle |

As seen above, the model contains six convolutional layers followed by three fully connected layers and a regressor that outputs a predicted steering angle. The input of the first fully connected layer are flattened and concatenated versions of the output of the fifth and sixth convolutional layers, this allows the fully connected layers to use higher and lower level features from the convolutional layers which was found to considerably reduce the error when the model was being trained.

'Relu' activation was also added in each layer in order introduce non-linearity in the model. 

This model is almost identical to model developed and published by Nvidia which can be seen [here](https://devblogs.nvidia.com/deep-learning-self-driving-cars/). The main differance is the model in this project concatenates flattened versions of the fifth and sixth convolutional layers as mentioned above.

Prior to passing images to the NN, images are first normalized using Keras lambda function and then cropped using Keras Cropping2D function. 

#### 2. Solution Design Approach

Different models were tested and enhanced prior to reaching the final model architecture described above. Initially, a simple model with two convolutional layers and a signle fully connected layer was tested. While the model was able to successfully drive the car for a short time, it performed very poorly at certain turns where the lane lines are not very clear. Thus the number of layers and the depth of layers were incrementally increased until finally this [model](https://devblogs.nvidia.com/deep-learning-self-driving-cars/) from Nvidia was adopted. While a significant improvement was achieved, the model did not perform as good on the second track of [Udacity's](www.udacity.com) simulation environment. By slightly changing the model so that the first fully connected layer uses the flattened output of the last two convolutional layers a long with some preprocessing of the training data (as describe in section.6 below), the error was considerably reduced and the model performed better in the both tracks.

#### 3. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting. Additionally, the model was trained, validated and then tested on different data sets to ensure that the model was not overfitting. 

Finally, the model was also tested by running it through Udacity's simulator and ensuring that the vehicle could stay on the track.

#### 4. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually. A batch size of 32 was used and the model was trained with 10 epochs as the model started overfitting after that.

#### 5. Training data

The training data was obtained using the Udacity simulation environment in which I manually drove while recorded and the corresponding steering angles were recorded. The data used was acquired from driving on two different tracks several time. The figure below shows some sample training images along with their recorded steering angles:

![alt_text][image1]

Although the simulator provides images from three different cameras, one in the center, one on the right and another on the left. When running the Udacity simulator on autonomous mode, only cameras from the center image are passed to the NN model, thus the model must be trained to accept only a single image.

One possible solution was to utilize a constant correcting factor on the steering angles of images obtained from left/right cameras and pass them to the model as center images when training the NN. However, in my project, only data obtained from the center camera was utilized in training the model. That is because the driving behavior I followed when obtaining data was different for the two tracks since I tried to stick to one lane of a two lane road in the second track, while the first track consisted of a single wide lane in which I tried to keep in the middle. This meant that for the same displacement to the right or to the left, the car must perform differently to follow the desired behavior. This means that no single correction factor can be used to correct for right and left camera images from both tracks. Unfortunately, since I did not take this into account when collecting the training data, I did not implement any mechanism to be able to separate data from different tracks and assign a different correcting factor according to the track. In order to compensate for this shortcoming a huge data set was collected and only center images were utilized. Additionally, more data points were obtained by flipping the training images and taking the negative of their steering angles. This step doubled the number of available data points. 

The data was loaded using a generator function in order to avoid overloading the memory and the model was trained using the fit_generator function in Keras.

#### 6. Balancing training data

After all the steps described above were performed, the model was tested on both tracks. While autonomous driving using the obtained model was capable of clearing the tracks at low speeds, it was apparent that the car sometimes took too long to start turning, causing the vehicle to get of the road at higher speeds. It was apparent that the vehicle tends to drive at small steering angles in general, this becomes understandable when looking more closely at the dataset used to train the model:

![alt_text][image2]

As seen clearly in the previous bar plot, most of the data are for small steering angles. This might cause the model to tend to underestimate the actual required steering angle. In order to obtain a more balanced dataset, random data points from over-represented steering angle ranges were deleted from the data set while images were repeated for under-represented steering angles in order to obtain a more balanced representation of the full range of steering angles, the final data used for training the model is shown here:

![alt_text][image3]


This have significantly improved the performance of the model in autonomous mode.

## Final Results

Finally, the data set was also split into sets of 60% training, 20% validation and 20% testing. The exact size of each set is:
- Size of the training set is: 32679
- Size of the validation set is: 10893
- Size of the testing set is: 10894

And the final mean squared error on each set was:
- Training set error: 0.0503
- Validation set error: 0.0389
- Testing set error: 0.0371

Videos of the model autonomously driving the vehicle in the simulation envirnment can be seen [here](./track1.mp4) and [here](./track2.mp4).
