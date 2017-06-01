# **Finding Lane Lines on the Road** 

[//]: # (Image References)
[scope]: ./figure/scope.png "Scope of line-detection"
[image1]: ./examples/grayscale.jpg "Grayscale"
[Rplane]: ./figure/R.png "R-plane image"
[Gplane]: ./figure/G.png "G-plane image"
[Bplane]: ./figure/B.png "B-plane image"
[RGplane]: ./figure/RG.png "RG avarage image"
[RGBplane]: ./figure/RGB.png "RG avarage image"
[CannyEdges]: ./figure/edges.png "Canny Edge image"
[Hough]: ./figure/hough_line_image.png "Hough-transformation image"
[MaskedEdge]: ./figure/masked_edges.png "Masked edges image"
[PredictedLines]: ./figure/predicted_line.png "Predicted lines"
[PredictedLinesImage]: ./figure/predicted_line_image.png "Predicted lines image"

## Reflection

### 1. Concept

Generally edge-detection images are very noisy, consequently Hough-transformation with them have instability.
My aim is improving the stability of line-detection with them.

### 2. Description of my pipeline

My pipeline consisted of 6 steps. 

#### Step1: normalize image size

Image size are different for every target video images.
As the first step, this pipeline unifies image-size as 1000x500.

#### Step2: grayscale

Target lines to detect are colored with White or Yellow in images.

Blue-plane doesn't affect the line-detection algorithm.
So grayscale is made from average of Red-plane and Green-plane.

![(R+G)/2 image][RGplane]
![COLOR_RGB2GRAY][RGBplane]

### Step3: edge detection

Similar to lessons, edge images are detected Canny-algorithm after bluring.

![CannyEdges][CannyEdges]

### Step4: Mask

Left and right side masks are defined each other for post Linear-Regression step.

Mask shapes are defined as pentagon based on positions where lines appear in 3 video images.

![scope in image][scope]

![MaskedEdgeImage]][MaskedEdge]

### Step5: Hough-transformation

Hough-transformation is executed twice for left and right line.

![HoughImage]][Hough]

### Step6: 2nd line-prediction

To replace Hough-transformation results with two solid lines,
this step breaks up lines into points and predicts a line again.

In this step, RANSAC algorithm, has a strong stability toward noisy input, is applied to line-prediction.

And plus, a IIR filter interpolates missing frame.
This IIR also reinforce stability of the result of the line-prediction.

![PredictedLines]][PredictedLines]
![PredictedLinesImage]][PredictedLinesImage]


## 2. Identify potential shortcomings with your current pipeline

### line color issue

This pipeline assume lines to detect are colored White or Yellow, 
so in some country that uses blue lane-lines, it would not work well.

### heuristic filter issue

This pipeline uses some heuristic rule and filled many parameters, determined from only 3 videos.

The parameters require to tune every time for new input video.

### stabilizing method

This pipeline depends on some mathematical interpolation.

They can cause troubles under unexpected situations like crossover points, occlusions or sharp curves.

## 3. Suggest possible improvements to your pipeline

A possible improvement would be to add free space detection method.

Line detection would be naive because edge points information may be few and noisy.
Area sensing would be helpfull as its robustness.


Another potential improvement could be to complete shape model of lane-line.

At this challenge lane-line was assumed as solid-line, but actually they have some curvature factors.
Exact line shape model would make line prediction method easy and precisive.


---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report



---
