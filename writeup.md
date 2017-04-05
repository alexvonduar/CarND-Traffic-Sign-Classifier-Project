**Traffic Sign Recognition** 
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

[image1]: ./report/training_distribution.png "Original Distribution"
[image2]: ./report/aug_14.png "Type 14: Stop"
[image3]: ./report/aug_18.png "Type 18: General caution"
[image4]: ./report/aug_26.png "Type 26: Traffic signals"
[image5]: ./report/training_distribution_aug.png "Augmented Distribution"
[image6]: ./report/8_before.png "Speed Limit (120km/h)"
[image7]: ./report/10_before.png "No passing for vehicles over 3.5 metric tons"
[image8]: ./report/8_after.png "Speed Limit (120km/h)"
[image9]: ./report/10_after.png "No passing for vehicles over 3.5 metric tons"
[image10]: ./report/test_image.png "Test Images"
[image11]:	./report/no0.png "No. 1 "
[image12]:	./report/no1.png "No. 2 "
[image13]:	./report/no2.png "No. 3 "
[image14]:	./report/no3.png "No. 4 "
[image15]:	./report/no4.png "No. 5 "
[image16]:	./report/no5.png "No. 6 "
[image17]:	./report/no6.png "No. 7 "
[image18]:	./report/no7.png "No. 8 "
[image19]:	./report/no8.png "No. 9 "
[image20]:	./report/no9.png "No. 10"
[image21]:	./report/no10.png	 "No. 11"
[image22]:	./report/no11.png	 "No. 12"
[image23]:	./report/no12.png	 "No. 13"
[image24]:	./report/no13.png	 "No. 14"
[image25]:	./report/no14.png	 "No. 15"
[image26]:	./report/no15.png	 "No. 16"
[image27]:	./report/no16.png	 "No. 17"
[image28]:	./report/no17.png	 "No. 18"
[image29]:	./report/no18.png	 "No. 19"
[image30]:	./report/no19.png	 "No. 20"
[image31]:	./report/no20.png	 "No. 21"
[image32]:	./report/no21.png	 "No. 22"
[image33]:	./report/no22.png	 "No. 23"
[image34]:	./report/no23.png	 "No. 24"
[image35]:	./report/no24.png	 "No. 25"
[image36]:	./report/no25.png	 "No. 26"
[image37]:	./report/no26.png	 "No. 27"
[image38]:	./report/no27.png	 "No. 28"
[image39]:	./report/no28.png	 "No. 29"
[image40]:	./report/no29.png	 "No. 30"
[image41]:	./report/no30.png	 "No. 31"
[image42]:	./report/no31.png	 "No. 32"
[image43]:	./report/no32.png	 "No. 33"
[image44]:	./report/no33.png	 "No. 34"
[image45]:	./report/no34.png	 "No. 35"
[image46]:	./report/no35.png	 "No. 36"
[image47]:	./report/no36.png	 "No. 37"
[image48]:	./report/no37.png	 "No. 38"
[image49]:	./report/no38.png	 "No. 39"
[image50]:	./report/no39.png	 "No. 40"
[image51]:	./report/no40.png	 "No. 41"
[image52]:	./report/no41.png	 "No. 42"
[image53]:	./report/no42.png	 "No. 43"
[image54]:	./report/no43.png	 "No. 44"
[image55]:	./report/no44.png	 "No. 45"
[image56]:	./report/no45.png	 "No. 46"
[image57]:	./report/no46.png	 "No. 47"

My project code could be download from [here](https://github.com/alexvonduar/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

---

**Training Data Analyse**

First I use following code to print the dimensions of training and test data:

```python
import numpy as np

n_train = len(X_train)

n_test = len(X_test)

image_shape = X_train[0].shape
image_depth = image_shape[2]

n_classes = np.unique(y_train).shape[0]

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Depth of data =", image_depth)
print("Number of classes =", n_classes)
``` 

and the results are:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

Then I use histogram chart to show the distribution of training data by classes
```python
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import shuffle
from mpl_toolkits.axes_grid1 import ImageGrid

def show_image(images, index):
    image = images[index].squeeze()
    plt.figure(figsize=(1,1))
    plt.imshow(image)
    print(y_train[index])
    
def show_single_image(image, index):
    print("Type ", index, trafficSignName(index))
    plt.figure(figsize=(1,1))
    plt.imshow(image)
    print(y_train[index])
    
def compare_images(src1, src2, index, src3 = None, src4 = None):
    print("Type ", index, trafficSignName(index))
    ncols = 2
    if src3 != None:
        ncols += 1
    if src4 != None:
        ncols += 1
    fig = plt.figure(1, (4., 4.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(1, ncols),  # creates 1xn grid of axes
                 axes_pad=0.1,  # pad between axes in inch.
                 )
    grid[0].imshow(src1)
    grid[1].imshow(src2)
    if src3 != None:
        grid[2].imshow(src3)
    if src4 != None:
        grid[3].imshow(src4)
    plt.show()

traffic_labels = pd.read_csv('signnames.csv', sep=',')

def trafficSignName(num):
    return traffic_labels[traffic_labels['ClassId']==num]['SignName'][num]

fig, ax = plt.subplots()

hist, bins, patches = ax.hist(y_train, n_classes)
print("hist ", hist)

ax.set_xlabel('Label Id')
ax.set_ylabel('Number of Images')
ax.set_title(r'Train Image Distribution')

fig.tight_layout()
plt.show()
```

From the result bar chart, I can see the distribution is highly biased.

![alt text][image1]

So I dicied to augment the training set to equalize the distribution.

```python
target_num = np.int(np.max(hist) * 3 / 2)
print("target ", target_num)

target_total = target_num * n_classes
print("target total", target_total)

shape = X_train.shape

X_aug = np.zeros((target_total, shape[1], shape[2], shape[3]), dtype= X_train.dtype)
y_aug = np.zeros((target_total), dtype= y_train.dtype)

X_aug[0:shape[0],:,:,:] = X_train
y_aug[0:shape[0]] = y_train

p = shape[0]
show_flag = np.zeros((n_classes), dtype=int)
for i in range(n_classes):
    n = np.count_nonzero(y_train == i)
    pad_n = np.int(target_num - n)
    
    l = np.where(y_train == i)
    l = shuffle(l[0])

    points = np.random.randint(low=0, high = 4, size=(pad_n, 3, 2))

    index = 0
    for j in range(pad_n):
        src = np.array([[32,32],[63,32],[32,63]])
        dst = src + points[j]

        warp_mat = cv2.getAffineTransform(np.float32(src), np.float32(dst))
        if index == n:
            index = 0
        image_expand = cv2.copyMakeBorder(X_train[l[index]], 32, 32, 32, 32, cv2.BORDER_REPLICATE)
        image_expand = cv2.warpAffine(image_expand,warp_mat, (96,96))
        image = image_expand[32:64,32:64,:]
        if show_flag[i] == 0:
            compare_images(X_train[l[index]], image, i)
            show_flag[i] = 1
        X_aug[p] = image
        y_aug[p] = i
        p += 1
        index += 1

show_num = 12
selected_pics = np.zeros([n_classes, show_num])

X_train = X_aug
y_train = y_aug

fig, ax = plt.subplots()
hist, bins, patches = ax.hist(y_train, n_classes)

ax.set_xlabel('Label Id')
ax.set_ylabel('Number of Images')
ax.set_title(r'Augmented Train Image Distribution')

fig.tight_layout()
plt.show()

for i in range(n_classes):
    l = np.where(y_train == i)
    l = shuffle(l[0])
    print("label[ ", i, "]:", hist[i], " name: ", trafficSignName(i))

    fig = plt.figure(1, (8., 8.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(1, show_num),  # creates 2x2 grid of axes
                 axes_pad=0.1,  # pad between axes in inch.
                 )
    for j in range(show_num):
        im = X_train[l[j]]
        selected_pics[i][j] = l[j]
        grid[j].imshow(im)
    plt.show()

n_train = len(X_train)

n_test = len(X_test)

image_shape = X_train[0].shape
image_depth = image_shape[2]

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Depth of data =", image_depth)
```
To avoid black background, I first padding the input 32x32 images to 96x96 by replicate the outlier lines.
Then use numpy random to generate random transformed points of left-top, right-top, left-bottom points and
apply cv2 affine transform. Here's some results:

![alt text][image2]

![alt text][image3]

![alt text][image4]

After doin so, final distribution bar chart is looks like this:

![alt text][image5]

---

**Preprocessing**

As when doing augmented, I still find that some images are very bright and some images are very dark even hardly to see some thing.
So I think I should do:
1. apply histogram equalization to each image, which will improve the image to avoid too dark or too bright
2. normalize image by subtrat mean of each image then divide each pixel by 255.0
After these preprocessing steps, image histogram will narrow down to [-1, 1], and also eliminate the variance of brightness

Before preprocessing:

![alt text][image6] ![alt text][image7]

After preprocessing:

![alt text][image8] ![alt text][image9]


code is:
```python
import cv2

def yuv_normalization(images):
    output = np.ndarray(images.shape, dtype=np.float64)
    for i, image in enumerate(images):
        img = np.float64(image)
        mean = np.mean(img[:,:,0], dtype = np.float64)
        img[:,:,0] -= mean
        img = img / 256.0
        output[i] = img
    return output

def rescale_normalized_yuv(image):
    image[:,:,0] = image[:,:,0] * 128 + 128
    image[:,:,1] = image[:,:,1] * 256
    image[:,:,2] = image[:,:,2] * 256
    return np.uint8(image)

def rgb2yuv(images):
    output = np.ndarray(images.shape, dtype=np.uint8)
    for i, image in enumerate(images):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        image[:,:,0] = cv2.equalizeHist(image[:,:,0])
        output[i] = image
    return output


X_train_yuv = rgb2yuv(X_train)
X_test_yuv = rgb2yuv(X_test)
X_valid_yuv = rgb2yuv(X_valid)

X_train_norm = yuv_normalization(X_train_yuv)
X_test_norm = yuv_normalization(X_test_yuv)
X_valid_norm = yuv_normalization(X_valid_yuv)
```

**Learning Pipeline**

The code for my final model is located in the sixth cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         	        	|     Description	        				                                    	  | 
|:-----------------------------:|:-----------------------------------------------------------------------------------:| 
| Input         	        	| 32x32x3 RGB image   						                                    	  | 
| Layer 0: Convolution 1x1     	| 1x1 stride, valid padding, outputs 32x32x64                                   	  |
| Layer 1: Convolution 5x5      | 1x1 stride, valid padding, outputs 28x28x128                                        |
| Layer 1: RELU				    |												                                      |
| Layer 1: Max pooling	        | 2x2 stride,  outputs 14x14x128 				                                      |
| Layer 2: Convolution 5x5      | 1x1 stride, valid padding, outputs 10x10x256                                        |
| Layer 2: RELU					|											                                      	  |
| Layer 2: Max pooling	        | 2x2 stride,  outputs 5x5x256 				                                          |
| Fully connected layer 1		| concatinate layer 1 and layer 2 outputs, inputs 14x14x128 + 5x5*256, outputs 512    |
| RELU                          |                                                                                     |
| Dropout                       | keep proportion 0.5                                                                 |
| Fully connected layer 2       | input 512, outpus 256                                                               |
| RELU                          |                                                                                     |
| Fully connected layer 3       | input 256, outpus 128                                                               |
| RELU                          |                                                                                     |
| Fully connected layer 4       | input 128, outpus 84                                                                |
| Softmax			        	|       									                                          |
|                               |                                                                                     |

**Train, Validate and Test the Mode**

The code for training, validating and test is located in the seventh cell of the Ipython notebook.
I use 20 epochs with 128 batch size to train the model, and set the learning rate to 0.001. I use valid set images to do cross validation and test set images to do test.

My final model results were:
* validation set accuracy of 0.966
* test set accuracy of 0.944

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 
 I choose LeNet lab's implementation as a start point. It's a convolution networks followed with fully connected layers.
 First I use orininal RGB images feed into the pipeline, make changes needed from LeNet lab to accept RGB image, and the result is around 85%.
 Then I tried to convert RGB images to gray scale images, since all the 43 images have different shape and don't use color to separate each other. I use histogram equalization on gray scale image too, since I found some images are badly taken thus too dark or too bright; and I also add a 1x1 convolution layer before first convolution layer to try to extract more features form 1 grayscale layer. All of them above, I got around 90%.
 Finally, inspired by LeCunn's [paper](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=2&cad=rja&uact=8&ved=0ahUKEwjKltrQ8onTAhUX3WMKHQcsDwEQFggsMAE&url=http%3A%2F%2Fwww.ntu.edu.sg%2Fhome%2Fegbhuang%2Fpdf%2FTraffic-Sign-Recognition-Using-ELM-CNN.pdf&usg=AFQjCNGJ-W-2zv6ZmI9MAcwDcrfnWsyBHg&sig2=JqpCpc3344fEji1-2UsM7A). I adopt the yuv form. First convert RGB image to YUV image, then do histogram equalization and normalization to Y channel, and keep U, V channel unchanged only scale its range from 0-255 to 0.0-1.0. After applied these, I got final results above.


**Test a Model on New Images**

I download 47 images from web, crop and scale the traffic signs to 32x32 pixels:

![alt text][image10]

I think the 21th, 22th, 33th, 36th and 43th images are hard to identify because they are covered or tilt to the image plane.

The code for making predictions on my final model is located in the 9th, 10th and 11th cell of the Ipython notebook.

The accuracy for the new images is 76% which is much lower compared with test set 94%. Here are the results of the prediction:

![alt text][image11]

![alt text][image12]

![alt text][image13]

![alt text][image14]

![alt text][image15]

![alt text][image16]

![alt text][image17]

![alt text][image18]

![alt text][image19]

![alt text][image20]

![alt text][image21]

![alt text][image22]

![alt text][image23]

![alt text][image24]

![alt text][image25]

![alt text][image26]

![alt text][image27]

![alt text][image28]

![alt text][image29]

![alt text][image30]

![alt text][image31]

![alt text][image32]

![alt text][image33]

![alt text][image34]

![alt text][image35]

![alt text][image36]

![alt text][image37]

![alt text][image38]

![alt text][image39]

![alt text][image40]

![alt text][image41]

![alt text][image42]

![alt text][image43]

![alt text][image44]

![alt text][image45]

![alt text][image46]

![alt text][image47]

![alt text][image48]

![alt text][image49]

![alt text][image50]

![alt text][image51]

![alt text][image52]

![alt text][image53]

![alt text][image54]

![alt text][image55]

![alt text][image56]

![alt text][image57]

From the result above, we can see that this model have truble in distinct signs which have fine texture details and when the image tilt too much, it's not so efficient too.
