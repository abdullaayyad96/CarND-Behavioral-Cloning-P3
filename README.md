# Behaviorial Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

[image1]: ./report_figures/sample_run_Trim.gif "camera feedback"
[image2]: ./report_figures/sample_run_3p_Trim.gif "Sample run"
[image3]: ./report_figures/NN_architecture.emf "NN architecture"

Overview
---

In this project, an end to end deep neural network is developed using keras to clone the driving behavior of a human and autonomously steer a vehicle in a simulation envrironment. The model will output a steering angle to the autonomous vehicle based solely on the images fedback from a camera installed on the vehicle as shown in the below figures.

| Camera feedback         		|     ![alt_text][image1]	        					| 
|:--------------------------------------------:|:-------------:| 
| __Autonomous Operation__        | ![alt_text][image2]   							|  

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior 
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
* Summarize the results with a written report


Simulator
---

[Udacity](www.udacity.com) provides a simulator as part of its [Self Driving Car NanoDegree Program](https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013) where a person can steer a car around a track for data collection. The simulator logs image data and steering angles which are used to train the neural network which is later used to drive the car autonomously around the track.

Dependencies
---
* Python3 
* Tensorflow=1.3.0
* keras==2.0.9
* NumPy
* OpenCV
* Matplotlib

All the required packages can be found in Udacity's CarND Term1 conda environment. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.


## Usage

### `drive.py`

Usage of `drive.py` requires you have saved the trained model as an h5 file, i.e. `model.h5`. The jupyter notebook [model.ipynb](model.ipynb) contains the code for developing the NN model and generating the h5 file.

Once the model has been saved, it can be used with drive.py using this command:

```sh
python drive.py model.h5
```

The above command will load the trained model and use the model to make predictions on individual images in real-time and send the predicted angle back to the simulator via a websocket connection.

Note: There is known local system's setting issue with replacing "," with "." when using drive.py. When this happens it can make predicted steering values clipped to max/min values. If this occurs, a known fix for this is to add "export LANG=en_US.utf8" to the bashrc file.

#### Saving a video of the autonomous agent

```sh
python drive.py model.h5 run1
```

The fourth argument, `run1`, is the directory in which to save the images seen by the agent. If the directory already exists, it'll be overwritten.

```sh
ls run1

[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_424.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_451.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_477.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_528.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_573.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_618.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_697.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_723.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_749.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_817.jpg
...
```

The image file name is a timestamp of when the image was seen. This information is used by `video.py` to create a chronological video of the agent driving. Sample images are provided in the `sample_run/` directory.

### `video.py`

```sh
python video.py run1
```

Creates a video based on images found in the `run1` directory. The name of the video will be the name of the directory followed by `'.mp4'`, so, in this case the video will be `run1.mp4`.

Optionally, one can specify the FPS (frames per second) of the video:

```sh
python video.py run1 --fps 48
```

Will run the video at 48 FPS. The default FPS is 60.

Running the script for the `sample_run/` directory generates this [video](./sample_run.mp4). 

Testing in simulator
---
The [track1.mp4](./track1.mp4) and [track2.mp4](./track2.mp4) demonstrate the deep learning model controlling the vehicle in two different tracks inside the simulation environment.

Neural Network Architecture
---
The NN used in this project utilizes multiple Convolutional layers follower by three fully connected layers and a regressor to output the steering angle. A simple illustration of the NN architecture can be seen below:
![alt_text][image3]

For more details regarding the neural network and the technical implementation, refer to the project's [technical writeup](technical_writeup.md) or [jupyter notebook](model.ipynb).
