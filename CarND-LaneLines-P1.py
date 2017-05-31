
# importing some useful packages
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.animation as animation
import numpy as np
import scipy as sc
from scipy import stats
import cv2

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
    return cv2.addWeighted(RGB[0], 0.5, RGB[1], 0.5, 0.0)

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
        # print(len(lines))
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
        # print(len(lines))
        for line in lines:
            for x1, y1, x2, y2 in line:
                length = math.hypot(x2 - x1, y2 - y1)
                theta = math.atan2((y2 - y1), (x2 - x1))
                degree = math.degrees(theta)
                # print(theta, degree, length)
                if 5 < length and  0.05*math.pi < theta and theta <  0.45*math.pi:
                    filtered_lines.append([[x1, y1, x2, y2]])
                    left_points.extend(devide_points(x1, y1, x2, y2))

                if 5 < length and -0.45*math.pi < theta and theta < -0.05*math.pi:
                    filtered_lines.append([[x1, y1, x2, y2]])
                    right_points.extend(devide_points(x1, y1, x2, y2))

    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, filtered_lines)

    # draw angle scale
    radial = 200
    for unit in (0.1, 0.2, 0.3, 0.4):
        rad = math.pi * unit
        cx = 500
        cy = 300
        x = int(math.sin(-rad) * radial + cx)
        y = int(math.cos(-rad) * radial + cy)
        cv2.line(line_img, (cx, cy), (x, y), [255, 0, 255], 1)
        cx = 550
        x = int(math.sin(rad) * radial + cx)
        y = int(math.cos(rad) * radial + cy)
        cv2.line(line_img, (cx, cy), (x, y), [0, 255, 255], 1)

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


def regression_line(img, left_points, color, thickness):

    # transform point list
    x = []
    y = []


    slope, intercept, r_value, _, _ = stats.linregress(x, y)

    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, filtered_lines)

    func = lambda x: x * slope + intercept
    line = lines.Line2D([0, 50], [func(0), func(50)])
    cv2.line(img, (x1, y1), (x2, y2), color, thickness)

    return line_img


def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)

    normalized_image = cv2.resize(image, (1000, 500))
    # print(normalized_image.shape)

    # Step1: create GrayScale and normalize size
    gray = grayscale(normalized_image)



    # Step2: Edge Detection
    blur_gray = gaussian_blur(gray, 5)
    # edges = canny(blur_gray, 50, 150)
    edges = canny(blur_gray, 0, 150)

    # Step3: Mask
    #imshape = image.shape
    imshape = normalized_image.shape
    vertices = np.array([[(100, 500),
                          (430, 300),
                          (570, 300),
                          (950, 500),
                          (800, 500),
                          (530, 320),
                          (470, 320),
                          (250, 500)]],
                        dtype=np.int32)
    masked_image = region_of_interest(edges, vertices)

    # Step4: Hough-transformation
    line_image, left_points, right_points = hough_lines(masked_image,
                             rho=10,            # 1 # distance resolution in pixels of the Hough grid
                             theta=np.pi / 120, # 180 # angular resolution in radians of the Hough grid
                             threshold=30,      # minimum number of votes (intersections in Hough grid cell)
                             min_line_len=3,    # 5 #minimum number of pixels making up a line
                             max_line_gap=1)    # 1 # maximum gap in pixels between connectable line segments

    # Step: devide lines into points
    
    line_image = regression_line(masked_image, left_points, [255, 0, 0], 8)
    print(left_points)
    # exit(0)

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
# for file in files:
#     image = mpimg.imread('test_images/' + file)
#     im = plt.imshow(process_image(image))
#     plt.savefig('test_images_output/tmp_' + file, pad_inches=0.0)
# exit(0)

# Test Images
# Build your pipeline to work on the images in the directory "test_images"
# You should make sure your pipeline works well on these images before you try the videos.
files = os.listdir("test_images/")
files = glob.glob("test_images/*.jpg")
files = glob.glob("test_images/challenge*.jpg")

fig = plt.figure()
ims = []
for file in files:
    # image = mpimg.imread('test_images/' + file)
    image = mpimg.imread(file)
    im = plt.imshow(process_image(image))
    ims.append([im])

ani = animation.ArtistAnimation(fig, ims, interval=200, blit=True, repeat_delay=0)
# ani.save('dynamic_images.mp4')
plt.show()
exit(0)



#  To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
#  To do so add .subclip(start_second,end_second) to the end of the line below
#  Where start_second and end_second are integer values representing the start and end of the subclip
#  You may also uncomment the following line for a subclip of the first 5 seconds
# clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)

white_output = 'test_videos_output/solidWhiteRight.mp4'
clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image)  # NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)


white_output = 'test_videos_output/solidYellowLeft.mp4'
clip1 = VideoFileClip("test_videos/solidYellowLeft.mp4")
white_clip = clip1.fl_image(process_image)  # NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)


white_output = 'test_videos_output/challenge.mp4'
clip1 = VideoFileClip("test_videos/challenge.mp4")
white_clip = clip1.fl_image(process_image)  # NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)
