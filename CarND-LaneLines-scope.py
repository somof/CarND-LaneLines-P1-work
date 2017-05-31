
# importing some useful packages
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.animation as animation
import numpy as np
import cv2

import glob
import scipy as sc
from scipy import stats
from sklearn import linear_model, datasets
import math

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML


# Helper Functions


def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    
    RGB = cv2.split(img)
    return RGB[0]
    # return cv2.addWeighted(RGB[0], 0.5, RGB[1], 0.5, 0.0)

    # return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """

    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)

    left_points = []
    right_points = []
    filtered_lines = []
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                length = math.hypot(x2 - x1, y2 - y1)
                theta = math.atan2((y2 - y1), (x2 - x1))
                degree = math.degrees(theta)
                if 5 < length and  0.05*math.pi < theta and theta <  0.45*math.pi:
                    filtered_lines.append([[x1, y1, x2, y2]])
                    right_points.extend(devide_points(x1, y1, x2, y2))

                if 5 < length and -0.45*math.pi < theta and theta < -0.05*math.pi:
                    filtered_lines.append([[x1, y1, x2, y2]])
                    left_points.extend(devide_points(x1, y1, x2, y2))

    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, filtered_lines)

    # draw angle scale
    # radial = 200
    # for unit in (0.1, 0.2, 0.3, 0.4):
    #     rad = math.pi * unit
    #     cx = 500
    #     cy = 300
    #     x = int(math.sin(-rad) * radial + cx)
    #     y = int(math.cos(-rad) * radial + cy)
    #     cv2.line(line_img, (cx, cy), (x, y), [255, 0, 255], 1)
    #     cx = 550
    #     x = int(math.sin(rad) * radial + cx)
    #     y = int(math.cos(rad) * radial + cy)
    #     cv2.line(line_img, (cx, cy), (x, y), [0, 255, 255], 1)

    return line_img, left_points, right_points

# Python 3 has support for cool math symbols.


def weighted_img(img, initial_img, α=0.8, β=1.0, λ=0.0):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)


# reading in an image
# image = mpimg.imread('test_images/solidWhiteRight.jpg')

# printing out some stats and plotting
# print('This image is:', type(image), 'with dimensions:', image.shape)
# plt.imshow(image)  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')

# Ideas for Lane Detection Pipeline
# Some OpenCV functions (beyond those introduced in the lesson) that might be useful for this project are:
# cv2.inRange() for color selection
# cv2.fillPoly() for regions selection
# cv2.line() to draw lines on an image given endpoints
# cv2.addWeighted() to coadd / overlay two images cv2.cvtColor() to grayscale or change color cv2.imwrite() to output images to file
# cv2.bitwise_and() to apply a mask to an image
# Check out the OpenCV documentation to learn about these and discover even more awesome functionality!

# TODO: Build your pipeline that will draw lane lines on the test_images
# then save them to the test_images directory.


def devide_points(x1, y1, x2, y2):
    length = math.hypot(x2 - x1, y2 - y1)
    num = int(math.sqrt(length))
    lines = []
    for i in range(0, num):
        lines.append([x1 + i * (x2 - x1) / num,
                      y1 + i * (y2 - y1) / num])
    return lines


def regression_line(img, points, hrange, color, thickness):

    # transform point list
    x = [d[0] for d in points]
    y = [d[1] for d in points]

    # Linear regressor
    # slope, intercept, r_value, _, _ = stats.linregress(x, y)
    # # func = lambda x: x * slope + intercept
    # func = lambda y: int((y - intercept) / slope)
    # cv2.line(img, (func(500), 500), (func(300), 300), color, thickness)

    if 30 < len(x):
        # RANSAC regressor
        model_ransac = linear_model.RANSACRegressor(linear_model.LinearRegression(),
                                                    min_samples=24,
                                                    residual_threshold=15,
                                                    # is_data_valid=is_data_valid,
                                                    # random_state=0
                                                    )
        if model_ransac is not None:
            X = np.array(x)
            # X = x[:, np.newaxis]
            model_ransac.fit(X[:, np.newaxis], y)
            # inlier_mask = model_ransac.inlier_mask_
            # outlier_mask = np.logical_not(inlier_mask)
        
            # line_x = np.arange(100, 401, 300)
            line_x = np.arange(hrange[0], hrange[1], hrange[1] - hrange[0] - 1)
            line_y = model_ransac.predict(line_x[:, np.newaxis])
            # print(line_x)
            # print(line_y)
        
        
            # func = lambda x: x * slope + intercept
            # cv2.line(img,
            #          (int(line_x[0]), int(line_y[0])),
            #          (int(line_x[1]), int(line_y[1])),
            #          color, thickness)

            return img, (int(line_x[0]), int(line_y[0])), (int(line_x[1]), int(line_y[1]))

    return img, [0, 0], [0, 0]


def process_image(image, weight=0.2):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)

    normalized_image = cv2.resize(image, (1000, 500))
    # print(normalized_image.shape)

    # Step0: Scope of line-detection
    scope = np.array([[(100, 500),
                       (430, 300),
                       (570, 300),
                       (970, 500),
                       (800, 500),
                       (520, 320),
                       (490, 320),
                       (250, 500)]],
                       dtype=np.int32)
    # scope = scope.reshape((-1,1,2))
    # img = cv2.polylines(normalized_image, [scope], True, (0,255,255))
    # return img


    # Step1: create GrayScale and normalize size
    gray = grayscale(normalized_image)
    # RGB = cv2.split(image)
    # gray = RGB[0]
    # gray_image= np.dstack((gray, gray, gray))
    # return gray_image


    # Step2: Edge Detection
    blur_gray = gaussian_blur(gray, 5)
    edges = canny(blur_gray, 0, 150)

    # Step3: Mask
    masked_image = region_of_interest(edges, scope)

    # Step4: Hough-transformation
    line_image, left_points, right_points = hough_lines(masked_image,
                                                        rho=10,            # 1 # distance resolution in pixels of the Hough grid
                                                        theta=np.pi / 120, # 180 # angular resolution in radians of the Hough grid
                                                        threshold=30,      # minimum number of votes (intersections in Hough grid cell)
                                                        min_line_len=3,    # 5 #minimum number of pixels making up a line
                                                        max_line_gap=1)    # 1 # maximum gap in pixels between connectable line segments

    # Step5: Linear regression
    line_image, lp0, lp1 = regression_line(line_image, left_points,  [100, 451], [255, 200, 0], 8)
    line_image, rp0, rp1 = regression_line(line_image, right_points, [550, 901], [200, 255, 0], 8)

    # step6: Infinite Impulse Response filter
    global lsp0, lsp1, rsp0, rsp1
    if lsp0[0] == 0 and lsp0[1] == 0:
        lsp0 = list(lp0)
    if rsp0[0] == 0 and rsp0[1] == 0:
        rsp0 = list(rp0)
    if lsp1[0] == 0 and lsp1[1] == 0:
        lsp1 = list(lp1)
    if rsp1[0] == 0 and rsp1[1] == 0:
        rsp1 = list(rp1)

    if lp0[0] != 0 or lp0[1] != 0:
        lsp0[0] = weight * lsp0[0] + (1.0 - weight) * lp0[0]
        lsp0[1] = weight * lsp0[1] + (1.0 - weight) * lp0[1]
    if rp0[0] != 0 or rp0[1] != 0:
        rsp0[0] = weight * rsp0[0] + (1.0 - weight) * rp0[0]
        rsp0[1] = weight * rsp0[1] + (1.0 - weight) * rp0[1]

    if 0 < weight:
        cv2.line(line_image, (int(lsp0[0]), int(lsp0[1])), (int(lsp1[0]), int(lsp1[1])), [200, 100, 150], 20)
        cv2.line(line_image, (int(rsp0[0]), int(rsp0[1])), (int(rsp1[0]), int(rsp1[1])), [200, 100, 150], 20)


    # StepX: Overlay

    # Create a "color" binary image to combine with line image
    # color_edges = np.dstack((edges, edges, edges))
    # result = weighted_img(line_image, color_edges, α=0.8, β=1., λ=0.)
    # result = weighted_img(line_image, image, α=0.8, β=1., λ=0.)
    result = weighted_img(line_image, normalized_image, α=0.8, β=1., λ=0.)
    # colored_gray = np.dstack((blur_gray, blur_gray, blur_gray))
    # result = weighted_img(line_image, colored_gray, α=0.8, β=1., λ=0.)

    return result


# Fusibility Test
# files = ['solidWhiteCurve.jpg', 'solidWhiteRight.jpg', 'solidYellowCurve.jpg', 'solidYellowCurve2.jpg', 'solidYellowLeft.jpg', 'whiteCarLaneSwitch.jpg']
# lsp0 = [0, 0]
# lsp1 = [0, 0]
# rsp0 = [0, 0]
# rsp1 = [0, 0]
# for file in files:
#     image = mpimg.imread('test_images/' + file)
#     im = plt.imshow(process_image(image, weight=0.0))
#     plt.savefig('test_images_output/tmp_' + file, pad_inches=0.0)
# exit(0)

# Test Images
# Build your pipeline to work on the images in the directory "test_images"
# You should make sure your pipeline works well on these images before you try the videos.
files = os.listdir("test_images/")
files = glob.glob("test_images/challenge*.jpg")
files = glob.glob("test_images/*.jpg")
files = ['solidWhiteCurve.jpg', 'solidWhiteRight.jpg', 'solidYellowCurve.jpg', 'solidYellowCurve2.jpg', 'solidYellowLeft.jpg', 'whiteCarLaneSwitch.jpg', 'challenge_000001.jpg', 'challenge_000010.jpg', 'challenge_000020.jpg', 'solidWhiteRight_000001.jpg', 'solidWhiteRight_000010.jpg', 'solidWhiteRight_000019.jpg', 'solidYellowLeft_000001.jpg', 'solidYellowLeft_000010.jpg', 'solidYellowLeft_000020.jpg', 'solidYellowLeft_000030.jpg', 'solidYellowLeft_000040.jpg', 'solidYellowLeft_000050.jpg']

files = ['solidYellowCurve2.jpg']

fig = plt.figure()
ims = []

lsp0 = [0, 0]
lsp1 = [0, 0]
rsp0 = [0, 0]
rsp1 = [0, 0]

for file in files:
    image = mpimg.imread('test_images/' + file)
    # image = mpimg.imread(file)
    im = plt.imshow(process_image(image))
    ims.append([im])

ani = animation.ArtistAnimation(fig, ims, interval=200, blit=True, repeat_delay=0)
# ani.save('dynamic_images.mp4')
plt.show()
