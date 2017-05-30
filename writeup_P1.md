# **Finding Lane Lines on the Road** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file. But feel free to use some other method and submit a pdf if you prefer.

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps. 
First, I converted the images to grayscale, then I .... 

In order to draw a single line on the left and right lanes,
 I modified the draw_lines() function by ...

If you'd like to include images to show how the pipeline works, 
here is how to include an image: 



RGB



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

