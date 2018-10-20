
# **Self-Driving Car Engineer Nanodegree - Behavioral Cloning Project** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)
[image1]: ./images/cnn-architecture.png "Model Architecture"
[image2]: ./images/center_2018_10_09_03_25_19_220.jpg "Center Camera Image"
[image3]: ./images/left_2018_10_09_03_25_19_220.jpg "Left Camera Image"
[image4]: ./images/right_2018_10_09_03_25_19_220.jpg "Right Camera Image"
[image5]: ./images/center-2017-02-06-16-20-04-855.jpg "Normal Image"
[image6]: ./images/center-2017-02-06-16-20-04-855-flipped.jpg "Flipped Image"
[image7]: ./images/normal_brightness.jpg "Normal Brightness"
[image8]: ./images/random_brightness.jpg "Random Brightness"

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is based on [nVidia model](https://devblogs.nvidia.com/deep-learning-self-driving-cars/) which has been proven to work in this problem domain. I just add some preprocessing layers as normalization layer dividing by 255 and discount 0.5 from input, cropping layer cutting 70 pixels from top and 25 from the bottom and resizing layer to (66,200,3).

The function that build the model can the find in model.py line 67

The diagram below is a depiction of the nVidia model architecture:
![alt text][image1]

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layer in order to reduce overfitting (model.py lines 83). 

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Data augmentation

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road applying angle correction of 0.25

![alt text][image2]
![alt text][image3]
![alt text][image4]

I also increased the data set flipping the images horizontally

![alt text][image5]
![alt text][image6]

and after I applied random brightness for all images added before (model.py line 9).

![alt text][image7]
![alt text][image8]

#### 4. Training options

I splitted the data set into training and validation set using the size of 80/20 and I also shuffle the data set.
The number of epochs is 3 and the batch size is 32. The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 104).

### Tests

Total number of samples for each set is:
Train on 31728 samples, validate on 7932 samples

Epochs results:

Epoch 1/3
31728/31728 [==============================] - 85s 3ms/step - loss: 0.0205 - val_loss: 0.0188

Epoch 2/3
31728/31728 [==============================] - 43s 1ms/step - loss: 0.0149 - val_loss: 0.0204

Epoch 3/3
31728/31728 [==============================] - 42s 1ms/step - loss: 0.0132 - val_loss: 0.0161


In the first track the car stayed in the center all over the track, the result was very good. But on the second track the car did not stay in the center and after a while hit the wall. Despite having applied a random brightness, it was not enough to drive well on a track that no examples were collected, but maybe adding more images with shadows or something may be improve this.

