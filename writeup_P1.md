# **Finding Lane Lines on the Road** 

[//]: # (Image References)
[scope]: ./figure/scope.png "Scope of line-detection"
[image1]: ./examples/grayscale.jpg "Grayscale"
[Rplane]: ./figure/R.png "R-plane image"
[Gplane]: ./figure/G.png "G-plane image"
[Bplane]: ./figure/B.png "B-plane image"
[RGplane]: ./figure/RG.png "RG avarage image"

## Reflection

### 1. Concept

Concept is Stability.
Generally, output of edge-detection is very noisy,
in spite of Lane-Detection for Automounous-Driving is based on it.

So I 
追記

2-Phase
1. careful selection of edge-candidates
2. predict lines from them

### 2. Description of my pipeline

My pipeline consisted of 5 steps. 

Step0: Scope of line-detection

![scope in image][scope]


左右に分ける
Hough変換も、ここから分ける


Step1: create GrayScale and normalize size

R or (R+G)/2

実際のところ、
このpipelineは cv2.COLOR_RGB2GRAY でも良く動作する


Step2: Edge Detection and Mask

adapt
Gaussian_Blur and Canny-Edge detection
as lesson

Step3: Mask
with Scope

Step4: Hough-transformation

adapt Hough-transformation with masked edge

For post Linear-Regression process,
detected lines are filtered again via with their angle

Output remaining edge-candidates as two list of points


Step5: Linear regression
Predict two lines from the list of points
with RANSAC algorithm.

Almost all of dashed-line would be identified as a solid-line.


step6: Infinite Impulse Response filter

Fill missing line-detections with previous line-detection's output.


Step7: Overlay image





RGB

コンセプト
確実な候補を残す
欠けた部分を補間する


Experimental Rule for line detection
- 白線は白か黄色 -> GrayScaleには RとGのみ使う
- 正常な運転の最中は、白線があるべき位置はおよそ決まる -> マスク領域でフィルタする
- 正常な運転の最中は、白線の画面上の取りうる角度には範囲がある -> Hough変換で検出した線分をフィルタする

Experimental Rule for solid-line detection
 prediction
かいき
revolution/recurrence/recursio
logistic
ゆるい曲線
自車の移動や他車で隠される以外の理由で、白線が突然消えたりしない
突然



まず、検証用の画像を用意
課題のビデオ3種類から、0.5秒おきに、静止画を抽出

画像大きさが異なるので、
処理を解こす前に、大きさを正規化しておく

黄色と白の車線なので
RGのみでGRAY化


前方の遠くのエッジが安定しないので除外



確実なエッジを残す

認識結果の数が減って、データの無い区間や時間が出来る分は
時間方向の補完で解決する




![alt text][image1]


### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when ... 

Another shortcoming could be ...

青い車線

空想区間が

交差点では使えない


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to ...

Another potential improvement could be to ...

### You can use this file as a template for your writeup if you want to submit it as a markdown file. But feel free to use some other method and submit a pdf if you prefer.

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report



---
