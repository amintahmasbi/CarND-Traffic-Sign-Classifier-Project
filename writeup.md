#**Traffic Sign Recognition** 


**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/amintahmasbi/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

[Check the notebook file]
I could use the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is ?
* The size of the validation set is ?
* The size of test set is ?
* The shape of a traffic sign image is ?
* The number of unique classes/labels in the data set is ?

####2. Include an exploratory visualization of the dataset.

[Under construction]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

**Answer:**

I have followed the preprocessing technique used in this paper (each image is converted to four(4) different training images):
**CireşAn, Dan, et al. "Multi-column deep neural network for traffic sign classification." Neural Networks 32 (2012): 333-338.**
- _Image Adjustment_: increases image contrast by mapping pixel intensities to new values such that 1% of the data is saturated at low and high intensities
- _Histogram Equalization_: enhances contrast by transforming pixel intensities such that the output image histogram is roughly uniform
- _Adaptive Histogram Equalization_: operates on tiles rather than the entire image: the image is tiled in 4 nonoverlapping regions of 8x8 pixels each. Every tile’s contrast is enhanced such that its histogram becomes roughly uniform
- _Contrast Normalization_: enhances edges through filtering the input image by a difference of Gaussians. We use a filter size of 5x5 pixels


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

### **Answer:**

Each neuron’s activation function is a hyperbolic tangent instead of relu. I also did not use dropout. Here is the final architecture of my CNN:

**Layer** | **Type** | ** # maps & neurons** | **Kernel**
:---: | :---: | :---: |:---:
0|input|3 maps of 32x32 neurons| 
1|convolutional|100 maps of 30x30 neurons| 3x3
2|max pooling|100 maps of 15x15 neurons| 2x2
3|convolutional|150 maps of 12x12 neurons| 4x4
4|max pooling|150 maps of 6x6 neurons| 2x2
5|convolutional|250 maps of 4x4 neurons| 3x3
6|max pooling|250 maps of 2x2 neurons| 2x2
7|fully connected|300 neurons| 1x1
8|fully connected|43 neurons| 1x1

####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

**Answer:**

* I tried the **learning rates** of $0.001$ and $0.0001$ and the latter one gave better results and smoother convergence. 
* The **EPOCHS** is set to $30$ because of smaller learning rate. 
* The **batch size** is kept as $128$, similar ot 5-LeNet implementation. 
* I used **AdamOptimizer** and the cost function was **softmax cross entropy with logits**.
* **Initial weights** are drawn from a truncated gaussian random distribution in the range of $[-0.5,0.5]$.

There are no other hyperparameters.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

**Answer:**

As mentioned above, I have picked a model based on a paper which has the best known performance on the German traffic sign dataset. The paper trained a couple of CNNs (in a multi-column structure) and averaged their outputs. I only implemented one network and merged all the training set into one big set (a single normalized dataset was also tested and did not give acceptable results). 
Finally, after tweaking few parameters such as _learning rate_ and _EPOCHS_ the performance of the network on validation set became $99.8$%. I tested the network on two test sets, one seperated from the training set and the actual test set in the pickle file and I got $98.1$% of performance on the latter which was satisfactory and close to paper outputs.
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

**Answer:**

The shape and design and U.S. signs are generally different from Europe and Germany. So, it is expected that the classification cannot detect those signs correctly. The images are plotted above.
I aslo put a sign with no label in the list "NO LEFT TURN" and set its label as (-1) to check the uncertainity of the network.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

**Answer:**

No, it could not. The model could only detect the "speed limit (30)" sign and the "stop" sign. The performance is $33$%.
The visualization of the softmax probabilities is shown below and shows the uncertainity of the model.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

**Answer:**
(Check the notebook file)
Only in one of the predictions the second choice was the correct answer. The "Do not enter" sign.
When the sign has never been seen before (i.e., "NO LEFT TURN"), the model is really uncertain about the output.
Same thing is true about U.S.-based "pedestrian" sign (the shape is unfamiliar for classifier), so it is uncertain about the output.


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

[Under construction]


###4. _Describe how you set up the training, validation and testing data for your model. **Optional**: If you generated additional data, how did you generate the data? Why did you generate the data? What are the differences in the new dataset (with generated data) from the original dataset?_
**Answer:**
First, I shuffled the training datasets to avoid any bias from order of the inputs. 
Second, the dataset is divided into training/validation/testing based on 70/15/15-percent rule.

**_Optional_**: Based on the paper mentioned above, for all of 4 different preporcessed images, I created three distorted image. The amount of distortions  are stochastic and applied to each preprocessed image for training, using bounded values for translation, rotation and scaling. These values are drawn from a uniform distribution in a specified range, i.e. ±10% of the image size for translation, [0.9−1.1] for scaling and ±5 degree for rotation. These three images and their original preprocessed image, are then added to the training set.

I tried training and evaluating the below network without these distortions and additional and the perfomance improved around $1.5$% after adding them.
