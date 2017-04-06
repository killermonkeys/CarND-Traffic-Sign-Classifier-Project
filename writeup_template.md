# **Traffic Sign Recognition** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[signs-explore]: ./writeup/signs-explore.png "A random collection of signs"
[label-histogram]: ./writeup/label-histogram.png "A random collection of signs"
[examples]: ./writeup/examples.png "Examples from the web"
[softmax-scores]: ./writeup/softmax-scores.png "Softmax scores"
[9-image]: ./writeup/9_image.png "60 km/h image"
[9-featuremap]: ./writeup/9_featuremap.png "60 km/h featuremap"
[10-image]: ./writeup/10_image.png "ahead only image"
[10-featuremap]: ./writeup/10_featuremap.png "ahead only featuremap"
[6-image]: ./writeup/6_image.png "end of speed limit image"
[6-featuremap]: ./writeup/6_featuremap.png "60 km/h featuremap"
[12-image]: ./writeup/12_image.png "60 km/h image"
[12-featuremap]: ./writeup/12_featuremap.png "60 km/h featuremap"


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/killermonkeys/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 104396 (the original set plus two additional random rotations of each image)
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3 (converted to grayscale)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third code cell of the IPython notebook.  

I reviewed the signs to understand how they were formatted and whether I could see any clear issues with them.

![alt text][signs-explore]

I also looked at the distribution of training inputs by label. Here is a histogram which shows that some classes (such as class 0) are fairly unrepresented.

![alt text][label-histogram]


### Design and Test a Model Architecture

#### 1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the fourth code cell of the IPython notebook.

I spent quite a lot of time looking at different preprocessing steps. 

The first thing I explored was the color space. Initially I found much better results with color than grayscale, which was surprising. However much later after I increased the number of training examples, I retried with grayscale and it was only marginally worse and trained more quickly.

I then normalized the image using a histogram equalization from skimage. This was the most effective normalization I could find that was consistent across the data set.

So in the end:
1. Add two copies of rotatations (-10 to 10 degrees, random)
2. Convert to grayscale
3. Global histogram normalization

#### 2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

I used the existing split of training, validation, and testing data. I did not augment the validation or testing data because I was not sure if that was appropriate given that the validation and testing data was "real" and my augmentations were synthetic data. 

The main steps I settled on were to create more training examples by randomly rotating the image -10 to 10 degrees and adding 2 examples of this. I did this because in looking at the data set, many images showed a perspective skew. I had a very hard time making an effective de-skewing step because it was quite easy to distort the relative positions of the shape even more if I skewed in the wrong direction. So I decided to rotate the image, which keeps the same angles between lines but shows them in different orientations.

I did not transpose the image because I did not see much tranposition in the training data, so I assumed they were reasonably well-cropped ROIs.


#### 3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the seventh cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image   					| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 32x32x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x32 				|
| Convolution 5x5     	| 1x1 stride, same padding, outputs 16x16x64 	|
| RELU					|												|
| Max pooling			| 2x2 stride,  outputs 8x8x64 					|
| Flatten				| outputs 4096x									|
| Fully connected		| inputs 4096x, outputs 1024x					|
| RELU					|												|
| Dropout				|												|
| Fully connected		| inputs 1024x, outputs 43x						|
| Softmax				| 												|
 

I did a lot of experimentation with the model architecture. I initially started increasing the model size dramatically because I inferred that since my input was going to be larger (RGB) and there were more classes, the model would have to increase in size. However, my model became too big, so I dropped one of the FC layers, and reduced the depth of the convolutional layers. I then saw that my model got high values on the training set but did not do well in testing, so I added a dropout layer to try to increase resiliency. Finally I did some refactoring to be able to visualize the model.


#### 4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the eigth cell of the ipython notebook. 

My model training is basically the same as what I implemented for LeNet, based on the suggestions in that lab. I used the ADAM optimizer, and eventually used 0.0002 learning rate, which seemed to be very effective over 40-50 epochs. Larger learning rates seemed to bounce the validation accuracy around too much, though I stopped tweaking the rate after I started working on preprocessing and augmenting the data set so I might be able to improve still.


#### 5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:
* training set accuracy of not sure, I didn't ever print training set accuracy. 
* validation set accuracy of 0.958
* test set accuracy of 0.953

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?

I started with the LeNet architecture based on the fact that in many respects the problems are similar. However I initally believed the color values would be relevant so I immediately adapted it to support color channels and the 43 class labels.

* What were some problems with the initial architecture?

The initial architecture had low accuracy and was extremely slow to train because it was very large. I also believed that it was not have enough depth in the convolution layers to support 43 classes nor enough nodes in the FC layers, so despite these issues I made the model bigger.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

I made the model much larger at first, going to 3 convolutional layers with depth 128, and 3 FC layers, which were 8192x to 1024x at their largest. This was much much too large to train on my machine without a GPU and I was not able to get the EC2 GPU to work (later I did) so training took over 1 hr for 50 epochs with only the original set.

After this I researched more and decided to make the model radically smaller and focus on augmenting the training set, which was more likely to result in better marginal performance. I went to roughly the architecture you see here, but with RGB color.

Then I focused on reducing my learning rate from 0.01 to 0.002, because I believed that my rate was causing me to overshoot. I was also not sure that my images were normalized correctly, because I was normalizing them between 0-1 and not -1 to 1 (I'm still not sure). This made me concerned my initial gradient position in the first layers cause too much overshooting.

The note here regarding training set accuracy vs. validation set accuracy is interesting, I hadn't really thought of looking at training accuracy. Unfortunately I committed the cardinal sin and ran on my testing data multiple times. I did see overfitting, which is why I added a dropout layer. This was extremely effective and almost equalized the two accuracies.

* Which parameters were tuned? How were they adjusted and why?

I changed network layer sizes and architectures, epochs, and learning rate. I ran probably 50+ trainings in total, adjusting them for reasons above.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

The two convolutional layers are to get shape features such as curves, diagonals, numerals, and then cross those into larger elements.

The FC layers are to represent the relationship of each of those individual features and to map those on to classes. The dropout ensures that the network does not overfit on the training data and that there are a diverse set of features that represent each class. I believe the dropout was particularly important given that my synthetic data was very close to the existing data, which increased the likelihood of overfitting.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][examples]
(the labels above are the correct labels, not predictions)

I got nearly all these videos from youtube, from dashcams.

As such, these videos are typically taken from the center of the road, and I generally took the screenshot when the sign was close. That means there is more perspective skew. Additionally some images are occluded, some were taken in bad lighting, either when the sun was in front of the sign with the sign in shadow, or in low light.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

My model predicted 22/23 signs correctly, 0.957 accuracy. 

The failure was on the 17: "No Entry" sign, which was predicted to be 9: "No passing". As you can see the two signs share many similar characteristics, they are circular with a bar element. However there are some pretty clear differences in color, and both had over 3000 examples in the training set. I suspect that if I included some color features or if the orientation was different in the failing example, it might have been correctly identified them. However, the accuracy of the second test set is nearly identical so it does indicate the model is performing reasonably well on real-world examples.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

![alt text][softmax-scores]

The scores show very strong skew toward the selected model in every case except for a few. I had to use a log scale to be able to visualize the scores at all. 

As you can see the incorrectly labeled 17 shows a high-ish second score for 9. Also notable is the 7: 100km/h limit, which shows a strong score for 1, 2, 4, 5, all of which are other speed limits. Otherwise, as expected the model strongly favors the highest scorer, given the number of epochs (50) and the size of the training set, I expected it to be this discriminatory.



### Visualize the Neural Network's State with Test Images

I used the output of my second convolution, after the ReLU but before the pooling (which seems to drop too much information to make nice pictures. The images I looked at were:
1. [3] 60 kph speed limit
2. [35] ahead only
3. [12] priority road
4. [18] general caution

![alt text][9-image]
![alt text][9-featuremap]
![alt text][10-image]
![alt text][10-featuremap]
![alt text][6-image]
![alt text][6-featuremap]
![alt text][12-image]
![alt text][12-featuremap]


3 and 35 have round outlines and the round shape is clear in many filters. There are also some filters which appear to respond to the 0 (which is common to multiple speed limit signs) such as filter_39. For 35, there's less of a strong emphasis on the arrow shape in any filter than I expected.

12 and 18 have straight sides, and have diagonals at different angles, because 12 is square at 45 degrees and 18 is a triangle. Many filters emphasize the outline of the shape of the sign, responding either to parallel diagonals (filter_30, filter_50) or to pairs of diagonals on one side (filter_42). The triangular shape appears to have a filter (filter_17, filter_41) that responds to it specifically.

Another way to look at it is the filter response of the same filter to many different inputs. For example, filter_10 responds strongly to the top half of circles, and therefore shows weaker response to 12 (which has a pointed top) and very weak response to 18. 
