
PYTHON = ../miniconda3/envs/carnd-term1/bin/python

UNAME = ${shell uname}
ifeq ($(UNAME),Darwin)
PYTHON = ../../../src/miniconda3/envs/carnd-term1/bin/python
endif

all:
	$(PYTHON) CarND-LaneLines-P1.py

extract:
	ffmpeg -i test_videos/challenge.mp4       -r 2 -f image2 test_images/challenge_%06d.jpg
#	ffmpeg -i test_videos/solidWhiteRight.mp4 -r 2 -f image2 test_images/solidWhiteRight_%06d.jpg
#	ffmpeg -i test_videos/solidYellowLeft.mp4 -r 2 -f image2 test_images/solidYellowLeft_%06d.jpg

