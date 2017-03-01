**Vehicle Detection Project**

The goal of the project is to detect cars in images and video streams.
Two approaches were evaluated:
* sliding window + HOG feature extractor + SVM classifier
* YOLO neural network.
Both produced reasonable results, but the sliding window is significantly slower and it is difficult to imagine how to
scale it. I will now explain both methods, starting with the YOLO network, and then the sliding window.

## Setup

If using CPU:
```
conda env create -f environment.yml
./download.sh
```
GPU:
```
conda env create -f environment-gpu.yml
./download.sh
```

## Yolo neural network

One of the most popular neural networks for object detection is [YOLO](https://pjreddie.com/darknet/yolo/). It can
recognize 20 classes, the 6th of which is a car. The full list of classes can be obtained [here](http://host.robots.ox.ac.uk:8080/pascal/VOC/voc2007/htmldoc/index.html), or in the constructor of the
`YoloObjectDetector` class, located in the `yolo_object_detector.py` file.
We can use its [.cfg](https://github.com/pjreddie/darknet/blob/master/cfg/yolo-voc.cfg) file to directly map it to Keras Sequential
model, and then serialize it and use the pretrained weights. This also makes it very easy to use transfer learning to
add objects like pedestrians, traffic signs and others in the future. The output of the YOLO network is a vector of
length 1470, which encodes the probability, confidence and bounding boxes for the recognized objects. We then extract
those from the encoded vector, using the properties described in Section 2: Unified Detection of the 
[original paper](https://arxiv.org/pdf/1506.02640.pdf), this happens in the `extract_boxes_from_yolo_output` method
of the `YoloObjectDetector` class. See below for model architecture

```
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to
====================================================================================================
convolution2d_1 (Convolution2D)  (None, 16, 448, 448)  448         convolution2d_input_1[0][0]
____________________________________________________________________________________________________
leakyrelu_1 (LeakyReLU)          (None, 16, 448, 448)  0           convolution2d_1[0][0]
____________________________________________________________________________________________________
maxpooling2d_1 (MaxPooling2D)    (None, 16, 224, 224)  0           leakyrelu_1[0][0]
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 32, 224, 224)  4640        maxpooling2d_1[0][0]
____________________________________________________________________________________________________
leakyrelu_2 (LeakyReLU)          (None, 32, 224, 224)  0           convolution2d_2[0][0]
____________________________________________________________________________________________________
maxpooling2d_2 (MaxPooling2D)    (None, 32, 112, 112)  0           leakyrelu_2[0][0]
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 64, 112, 112)  18496       maxpooling2d_2[0][0]
____________________________________________________________________________________________________
leakyrelu_3 (LeakyReLU)          (None, 64, 112, 112)  0           convolution2d_3[0][0]
____________________________________________________________________________________________________
maxpooling2d_3 (MaxPooling2D)    (None, 64, 56, 56)    0           leakyrelu_3[0][0]
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 128, 56, 56)   73856       maxpooling2d_3[0][0]
____________________________________________________________________________________________________
leakyrelu_4 (LeakyReLU)          (None, 128, 56, 56)   0           convolution2d_4[0][0]
____________________________________________________________________________________________________
maxpooling2d_4 (MaxPooling2D)    (None, 128, 28, 28)   0           leakyrelu_4[0][0]
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 256, 28, 28)   295168      maxpooling2d_4[0][0]
____________________________________________________________________________________________________
leakyrelu_5 (LeakyReLU)          (None, 256, 28, 28)   0           convolution2d_5[0][0]
____________________________________________________________________________________________________
maxpooling2d_5 (MaxPooling2D)    (None, 256, 14, 14)   0           leakyrelu_5[0][0]
____________________________________________________________________________________________________
convolution2d_6 (Convolution2D)  (None, 512, 14, 14)   1180160     maxpooling2d_5[0][0]
____________________________________________________________________________________________________
leakyrelu_6 (LeakyReLU)          (None, 512, 14, 14)   0           convolution2d_6[0][0]
____________________________________________________________________________________________________
maxpooling2d_6 (MaxPooling2D)    (None, 512, 7, 7)     0           leakyrelu_6[0][0]
____________________________________________________________________________________________________
convolution2d_7 (Convolution2D)  (None, 1024, 7, 7)    4719616     maxpooling2d_6[0][0]
____________________________________________________________________________________________________
leakyrelu_7 (LeakyReLU)          (None, 1024, 7, 7)    0           convolution2d_7[0][0]
____________________________________________________________________________________________________
convolution2d_8 (Convolution2D)  (None, 1024, 7, 7)    9438208     leakyrelu_7[0][0]
____________________________________________________________________________________________________
leakyrelu_8 (LeakyReLU)          (None, 1024, 7, 7)    0           convolution2d_8[0][0]
____________________________________________________________________________________________________
convolution2d_9 (Convolution2D)  (None, 1024, 7, 7)    9438208     leakyrelu_8[0][0]
____________________________________________________________________________________________________
leakyrelu_9 (LeakyReLU)          (None, 1024, 7, 7)    0           convolution2d_9[0][0]
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 50176)         0           leakyrelu_9[0][0]
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 256)           12845312    flatten_1[0][0]
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 4096)          1052672     dense_1[0][0]
____________________________________________________________________________________________________
leakyrelu_10 (LeakyReLU)         (None, 4096)          0           dense_2[0][0]
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 1470)          6022590     leakyrelu_10[0][0]
====================================================================================================
Total params: 45,089,374
Trainable params: 45,089,374
Non-trainable params: 0
```

We then use the `detect_in_cropped_image` 


[//]: # (Image References)
[dataset_examples]: ./examples/car_not_car.png
[hog_example]: ./output_images/hog_example.jpg
[test1]: ./output_images/test1.jpg
[test2]: ./output_images/test2.jpg
[test3]: ./output_images/test3.jpg
[test4]: ./output_images/test4.jpg
[test5]: ./output_images/test5.jpg
[test6]: ./output_images/test6.jpg

[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The main action is in `car_not_car_classifier.py` file. I start by loading the dataset in the `load_data` method of the
class `CarNotCarClassifier`, which in turn calls the private method `_get_X_y`, where a utility function
`extract_features_from_filenames` located in `utils.py` is used to extract features from image file names, which reads
each image and then calls `extract_features_from_images`, which extracts features for a single image using the
`extract_features_single_image`.

Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][dataset_examples]

In the `extract_features_single_image`, I use `skimage`'s function `hog`, spatial features which are simply flattened
version of the image and histogram colors to form a vector, which represents the features. Here is an example hog feature
of the Br channel of an image:

![alt text][hog_example]

####2. Explain how you settled on your final choice of HOG parameters.

I tried different combinations of the parameters, even tried running a genetic algorithm to find the best parameters
using http://deap.readthedocs.io/en/master/, where the individuals are a combination of parameters and their fitness
is the accuracy on the test set. This led to 99.12% accuracy on the test set, but there were a lot of false positives
in the image detection, so I manually selected the parameters in the end, through trial and error to be:
orientation - 9, pix_per_cell - 8, cell_per_block - 2.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM in the `train_from_scratch` method of the class `CarNotCarClassifier`. First I split the data
using a fixed seed to get reproducible results. Then I perform standardization on the training data only, since

> For example, preparing your data
> using normalization or standardization on the entire training dataset before learning would not
> be a valid test because the training dataset would have been influenced by the scale of the data
> in the test set.
> -- http://machinelearningmastery.com/automate-machine-learning-workflows-pipelines-python-scikit-learn/

After that, I train the linear SVM and log its accuracy on the test set.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search. How did you decide what
scales to search and how much to overlap windows?

I decided to search at the bottom half of the image, cutting the sky. I choose the scales to be 1.0, 1.33, 1.66, 2.0
and performed the search by computing the features for the whole cropped image, and then subsampling it. I choose these
values as a tradeoff between speed and precision. In the end, the search takes approximately 1FPS, which is still very
slow.

####2. Show some examples of test images to demonstrate how your pipeline is working.
What did you do to optimize the performance of your classifier?

I optimized the performance of the classifier by searching at an optimal number of scales, to retain reasonable
accuracy and speed. Here is how the classifier performs on the test images. As can be seen from the test images below
though, some price is paid, as one of the images missed the car, because the car in the image
was not at a suitable case. Also, there are still some small false positives.

![alt text][test1]

![alt text][test2]

![alt text][test3]

![alt text][test4]

![alt text][test5]

![alt text][test6]


---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](https://youtu.be/sJHEa_ej6hU). This is using YOLO.

####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and
some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video. From the positive detections I created a
heatmap, see `create_heatmap` in `utils.py`. I filtered false positives by thresholding that map to identify vehicle
positions in the `threshold_heatmap` method. I then used `scipy.ndimage.measurements.label()` to identify individual
blobs in the heatmap in the `get_cars` method. I then assumed each blob corresponded to a vehicle.  I constructed
bounding boxes to cover the area of each blob detected and then draw them on the image.

In the YOLO implementation, I just combine each blob from `scipy.ndimage.measurements.label()`, and the YOLO network
has very low false positive rate, so there I do not have additional heuristics and cheap tricks to filter false
positives.

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your
pipeline likely fail?  What could you do to make it more robust?

Potential problems:

* YOLO seems to detect all the cars correctly, but as with all neural networks, there is possibility to fool the
network, as described in [this paper](https://arxiv.org/abs/1412.1897)
* HOG + SVM is slow, and the number of windows for searching can grow too quickly. I am also not sure how would it
benefit from using GPU.
* All the car images which were used for training the HOG + SVM were taken from behind, so if there is a car in our
lane which is heading towards us, we won't detect it, which is a huge safety issue.

The video shown uses YOLO