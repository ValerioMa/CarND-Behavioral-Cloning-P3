#**Behavrioal Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./img/cnn-architecture.png "NVIDIA cnn architecture"
[image2]: ./img/center_driving.png "Center lane driving"
[image3]: ./img/left_driving.png "Driving on left side"
[image4]: ./img/right_driving.png "Driving on right side"


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submssion includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submssion code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model arcthiecture has been employed

My model consist of a convolutional neural network similar to the one build from NVIDIA, which consists of 9 layers, including a normalization layer, 5 convolutional layers, and 3 fully connected layers. The input image is split into YUV planes and passed to the network.

The normalization is hard coded using a Keras lambda layer (code line 125) and is not adjusted in the learning process. 

The first three convolutional layers use a 2x2 stride and 5x5 kernel, and the final two convolutional layers use a non-strided convolution with 3x3 kernel.

Depths of convolutional layer is between 32 and 64 (model.py lines 130-134).

The model includes RELU layers to introduce nonlinearity (code line 130-142), but the last fully connected layer have sigmoid as activation, this to try to provide a more smooth steering dynamic.

![alt text][image1]

####2. Attempts to reduce overfitting in the model

The convolutionla part doesn't have dropout layer but 3 dropout layers are inserted between the 3 fully connected layers of the classifier (code line 138-145)
The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 31-35). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 147).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, and side lane driving (left and right).

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to study some publications and in particular my solution is based on the one proposed in *End to End Learning for Self-Driving Cars* (arXiv:1604.07316). 

So my first step was to use a convolution neural network model similar to the one proposed in the NVIDIA paper; I thought this model might be appropriate because it is already used in real applications.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. 

Initially I trained the model only on the given data, but this was not enough, in fact there were few spots where the vehicle fell off the track (most of the time just before and after the bridge or in the last sharp turn to the right). 

I decided to create new data with the simulator using a Lofitec steering wheel that allow me to have mode smooth data.

Like in the NVIDIA paper I used the images from left and right cameras in addition to the center one. This had the purpose to teach the system how to stay in the center of the road. I have done that by adding a different steering offset to the side cameras.

This led to a weaving motion and so I add some images in which the car has an angle aproximately of 45° with the main direction of the street and the steering angle required to keep the car in the center of the road.
 
The final step was to run the simulator to see how well the car was driving around track one. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road, but still has some weaving behavior.

####2. Final Model Architecture

The final model architecture (model.py lines 123-146) consisted of a convolution neural network with the following layers and layer sizes:

- Normalization layer (weights are hard-coded);
- Convolution
	- 1st Convoluitonal Layer 24 features, 5x5 kernel, 2x2 stride, activation relu;
	- 2nd Convoluitonal Layer 36 features, 5x5 kernel, 2x2 stride, activation relu;
	- 3rd Convoluitonal Layer 48 features, 5x5 kernel, 2x2 stride, activation relu;
	- 4th Convoluitonal Layer 64 features, 3x3 kernel, 1x1 stride, activation relu;
	- 5th Convoluitonal Layer 64 features, 3x3 kernel, 1x1 stride, activation relu;
	- Flatten layer;
- Classification
	- 1st fully connected layer, 1164 nodes, activation relu;
	- dropout with probability 0.5
	- 2nd fully connected layer, 100 nodes, activation relu;
	- dropout with probability 0.5
	- 3rd fully connected layer, 50 nodes, activation relu;
	- dropout with probability 0.2
	- 4th fully connected layer, 10 nodes, activation sigmoid;
	- final layer 1 node.
	
I used the adam optimizer with mean squared error as loss function.

Here is a visualization of the architecture:

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded 4 laps (2 on the left side, and 2 on the right side) to teach the vehicle how to recover from the left side and right side of the road back to center. 
To collect the data I simply drove the car on the side (left or right) of the track and in pos-processing I added an offset simulating the car steering toward the street center:

![alt text][image3]
![alt text][image4]


After the collection process, I had 37000 number of data points. I then preprocessed this data by adjusting the steering angle;; cropping the image to remove the car (bottom) and the landscape (top) and converting the RGB image to YUV image.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting.
