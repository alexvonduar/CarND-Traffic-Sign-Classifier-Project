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
[image5]: ./report/training_distribution.png "Augmented Distribution"
[image6]: ./report/8_before.png "Speed Limit (120km/h)"
[image7]: ./report/10_before.png "No passing for vehicles over 3.5 metric tons"
[image8]: ./report/8_after.png "Speed Limit (120km/h)"
[image9]: ./report/10_after.png "No passing for vehicles over 3.5 metric tons"
[image10]: ./report/test_image.png "Test Images"

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
* test set accuracy of 0.940

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

The accuracy for the new images are 70%. Here are the results of the prediction:

|Image  | Sign Type			        | Prediction1 | Prediction2 | Prediction3 | Prediction4 | Prediction5 |
|:----:|:---------------------:|:-------------------------------------:|:-------------------------------------:|:-----------------------------:|:------------------------------:|:-------------------------------------:|
|96822779.bmp|Double curve|Double curve 21.0618|Road work 12.4938|Right-of-way at the next intersection 10.8621|Wild animals crossing 5.07889|Keep right 3.15149|
|5.bmp|Stop|Stop 33.3703|Bicycles crossing 4.27079|No vehicles 2.4282|Speed limit (30km/h) -0.213573|Speed limit (60km/h) -1.1572|
|3.bmp|Speed limit (120km/h)|Speed limit (20km/h) 3.5835|Speed limit (30km/h) 2.82731|Keep left 2.29803|No vehicles 1.91148|Speed limit (70km/h) 0.884721|
|688071187.bmp|Speed limit (60km/h)|Speed limit (60km/h) 29.7834|Speed limit (50km/h) 23.0325|Speed limit (80km/h) 11.0004|No vehicles 7.7941|Wild animals crossing -0.500433|
|171590640.bmp|Beware of ice/snow|Right-of-way at the next intersection 16.3356|Double curve 4.27044|End of no passing by vehicles over 3.5 metric tons 4.23512|Roundabout mandatory 3.24299|Slippery road 0.577272|
|689937191.bmp|No passing|No passing 15.9583|Dangerous curve to the left 11.2124|No vehicles 4.87325|No passing for vehicles over 3.5 metric tons 4.61441|Speed limit (80km/h) 3.56307|
|2.bmp|Pedestrians|Right-of-way at the next intersection 10.5009|General caution 7.15797|Road work 6.727|Beware of ice/snow 6.68699|Dangerous curve to the right 5.04717|
|142475664.bmp|General caution|General caution 57.1584|Right-of-way at the next intersection 19.4966|Traffic signals 15.9026|Stop 11.6353|Pedestrians 11.2965|
|6367584752.bmp|Road narrows on the right|Road narrows on the right 35.6783|Bicycles crossing 14.3032|Beware of ice/snow 9.89587|Pedestrians 9.7827|Road work 6.96623|
|459381359.bmp|No passing|No passing 53.5205|No passing for vehicles over 3.5 metric tons 19.1813|Vehicles over 3.5 metric tons prohibited 16.2193|No vehicles 15.9501|Speed limit (80km/h) 10.2206|
|106352829.bmp|Wild animals crossing|Wild animals crossing 61.2059|Road work 15.2528|Slippery road 12.2539|Bumpy road 8.54251|Double curve 7.45202|
|171209328.bmp|Stop|Stop 69.1038|Bicycles crossing 4.3732|No vehicles 2.30311|End of all speed and passing limits -2.97447|Turn right ahead -4.31351|
|689937195.bmp|Priority road|Priority road 96.8575|End of all speed and passing limits 3.56698|Stop -8.16243|Right-of-way at the next intersection -12.3169|Double curve -17.0272|
|469763319.bmp|Pedestrians|Pedestrians 21.5726|Right-of-way at the next intersection 14.7226|General caution 12.5558|Children crossing 8.6564|Dangerous curve to the left 3.64727|
|465649993.bmp|Speed limit (60km/h)|Speed limit (50km/h) 22.161|Speed limit (30km/h) 8.90783|Speed limit (70km/h) 5.01321|Speed limit (80km/h) 4.73603|Speed limit (100km/h) 2.50556|
|165186251.bmp|Beware of ice/snow|Beware of ice/snow 19.0871|Right-of-way at the next intersection 16.7961|Children crossing 3.55586|Slippery road 2.99843|Pedestrians 0.284532|
|57452573.bmp|Yield|End of all speed and passing limits 21.7993|End of no passing 3.53088|Turn left ahead 0.958656|Roundabout mandatory -0.00919942|End of speed limit (80km/h) -0.0676789|
|459381113.bmp|Speed limit (50km/h)|Speed limit (50km/h) 46.2037|Speed limit (30km/h) 22.7507|Speed limit (70km/h) 10.0719|No vehicles 5.19576|Speed limit (80km/h) 4.00983|
|532364055.bmp|No passing|No passing 74.1457|Dangerous curve to the left 25.6292|No passing for vehicles over 3.5 metric tons 12.8222|No vehicles 10.7884|Ahead only 7.15245|
|122911974.bmp|Priority road|Priority road 99.5719|Stop -5.34828|End of all speed and passing limits -7.65613|Bicycles crossing -8.63034|Right-of-way at the next intersection -16.375|
|649124149.bmp|Speed limit (30km/h)|Speed limit (100km/h) 5.51376|Speed limit (30km/h) 3.45241|Speed limit (80km/h) 2.38077|Speed limit (50km/h) 1.75675|Speed limit (120km/h) 1.65891|
|96150383.bmp|Speed limit (30km/h)|Speed limit (30km/h) 18.7706|Speed limit (20km/h) 13.8004|Speed limit (70km/h) 11.526|Speed limit (50km/h) 4.86421|Speed limit (80km/h) -0.159729|
|678876687.bmp|Wild animals crossing|Slippery road 5.82849|Beware of ice/snow 5.74751|Right-of-way at the next intersection 5.21141|Double curve 4.74657|Dangerous curve to the left 2.47593|
|646712504.bmp|Speed limit (50km/h)|Speed limit (50km/h) 26.11|Speed limit (30km/h) 10.5543|Speed limit (70km/h) 5.88813|Speed limit (80km/h) 4.72046|No vehicles 4.35316|
|636758475.bmp|Road work|Road work 56.5726|Double curve 8.53689|Beware of ice/snow 6.28407|Dangerous curve to the right 4.20553|Wild animals crossing 1.29207|
|469763309.bmp|Roundabout mandatory|Roundabout mandatory 43.384|End of speed limit (80km/h) 6.8977|Go straight or left 4.48019|Keep right 3.59518|Speed limit (30km/h) 3.2751|
|1063528292.bmp|30 Beware of ice/snow|Right-of-way at the next intersection 9.07139|Children crossing 8.16718|Pedestrians 7.72035|Beware of ice/snow 7.00531|Dangerous curve to the right 5.48678|
|139372530.bmp|Wild animals crossing|Wild animals crossing 18.9549|Bicycles crossing 7.42193|Double curve 6.84952|Speed limit (80km/h) 3.65153|Pedestrians 2.63037|
|6.bmp|Yield|Yield 59.29|Speed limit (60km/h) 10.6353|No vehicles 2.66863|Speed limit (80km/h) 2.20909|No passing 0.533554|
|122439599.bmp|Double curve|Double curve 53.3198|Wild animals crossing 16.5757|Speed limit (30km/h) 13.5758|limit (50km/h) 13.066|Speed limit (80km/h) 9.91273|
|153951752.bmp|Speed limit (50km/h)|limit (50km/h) 65.4802|Speed limit (30km/h) 35.7761|Speed limit (70km/h) 20.9007|Speed limit (80km/h) 11.6028|No vehicles 9.13664|
|125530335.bmp|General caution|General caution 24.247|Pedestrians 9.36503|Right-of-way at the next intersection 8.6905|Traffic signals 6.53023|Dangerous curve to the right 3.61672|
|656334061.bmp|Speed limit (30km/h)|Road work 4.56561|Turn right ahead 4.40654|Bicycles crossing 2.98922|Stop 2.90751|Priority road 2.6728|
|459381295.bmp|Children crossing|Children crossing 44.0761|Ahead only 13.2671|Dangerous curve to the left 7.76078|Bumpy road 6.77964|Pedestrians 5.63151|
|4.bmp|Speed limit (120km/h)|Speed limit (20km/h) 15.6229|Speed limit (120km/h) 7.96967|Children crossing 6.08955|General caution 3.36528|Traffic signals 3.03788|
|459380825.bmp|General caution|General caution 9.92734|Traffic signals 6.6202|Bumpy road 6.26291|Road narrows on the right 5.38707|Road work 5.11739|
|97447859.bmp|Speed limit (30km/h)|limit (50km/h) 19.969|Speed limit (30km/h) 17.2441|Speed limit (70km/h) 9.62397|Keep left 6.41679|Speed limit (80km/h) 5.11086|
|459381275.bmp|Right-of-way at the next intersection|Right-of-way at the next intersection 38.1386|Roundabout mandatory 7.92537|Beware of ice/snow 6.81134|General caution 1.39738|Double curve 0.674032|
|1.bmp|Road work|Road work 18.4649|Beware of ice/snow 4.58333|Road narrows on the right 4.35626|Wild animals crossing 1.57189|Dangerous curve to the right 0.941434|
|95909520.bmp|Road work|Road work 34.4235|limit (50km/h) 8.7344|Speed limit (80km/h) 6.20459|Speed limit (60km/h) 4.67218|Road narrows on the right 2.94636|
|674491693.bmp|No entry|No entry 46.6839|No passing 10.9115|Slippery road 10.7555|Vehicles over 3.5 metric tons prohibited 8.23244|Dangerous curve to the left 2.81226|
|1539517522.bmp|Double curve|Double curve 50.239|Right-of-way at the next intersection 34.4096|Road work 29.4411|Beware of ice/snow 28.5136|Wild animals crossing 10.1392|
|548310403.bmp|Yield|Yield 20.6534|No vehicles 8.06621|Speed limit (60km/h) 3.1423|Ahead only 2.74613|No passing 2.56263|
|607770600.bmp|Children crossing|Children crossing 89.7841|Ahead only 35.7093|Bicycles crossing 26.5204|Bumpy road 25.6216|Dangerous curve to the left 12.3576|
|688963145.bmp|Speed limit (100km/h)|Speed limit (100km/h) 19.3733|Speed limit (80km/h) 12.6116|Speed limit (120km/h) 6.94796|Vehicles over 3.5 metric tons prohibited 3.94948|limit (50km/h) 1.05437|
|142641434.bmp|Stop|Stop 26.7928|Bicycles crossing 4.09793|Road work 2.05886|No vehicles 0.254827|Turn right ahead -0.0894577|
|155907900.bmp|Speed limit (70km/h)|General caution 9.56075|Stop 7.4923|Speed limit (20km/h) 4.93237|No vehicles 3.69894|Road work 3.53795|

From the result above, we can see that this model have truble in distinct signs which have fine texture details, for example, "Right-of-way at the next intersection", some "Double Curve" and "Beware of ice/snow" are recognized as "Right-of-way at the next intersection", and when the image tilt too much, it's not so efficient too. Further more, I should add a RELU afer last full connected layer before softmax to avoid negative points.
