# **Behavioral Cloning** 

## Project Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/data_samples.jpg "Sample Images"
[image2]: ./figures/data_before_balancing.jpg "Data Bar Graph"
[image3]: ./examples/data+after_bar.jpg "Balanced Data Bar Graph"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission files

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup.md summarizing the project
* video.mp4 & video2.mp4 showing the performance of the trained model in two different tracks


#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.ipynb file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. Final NN Model Architecture

A discreption of the Neural Network model I built can be seen below:

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
| Concatentate  | Concatenate flattened output of  5'th and 6'th Convolutional layers
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

As seen above, the model contains six convolutional layers followed by three fully connected layers and a regressor that outputs a predicted steering angle. The input of the first fully connected layer are flattened and concatenated versions of the output of the fifth and sixth convolutional layers, this allows the fully connected layers to use higher and lower level features fron the convolutional layers which was found to considerabley reduce the error when the modelwas being trained.

'Relu' activation was also added in each layer in order introduce non-linearity in the system. 

This model is almost identical to model developed and published by Nvidia which can be seen [here] (https://devblogs.nvidia.com/deep-learning-self-driving-cars/). The main differance is the model in this project conctenates flatenned versions of the fifth ans sixth convolutional layers as mentioned above.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting. Additionally, the model was trained, validated and then tested on different data sets to ensure that the model was not overfitting. 

Finally, the model was also tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
