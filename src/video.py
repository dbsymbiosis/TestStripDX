import logging
import numpy as np
from moviepy import *
import moviepy.editor as mpy
from PIL import Image
import imageio.v2 as imageio



# Extract a single frame from a given time point in the provided video file
#
# Parameters:
#	video_filename	input video files
#	out_prefix	prefix to use for frames that we extract from video
#	seconds		second into video to grab frame from
def capture_frame(video_filename, out_prefix, seconds):
	seconds = sorted(seconds) # sort so it is ordered smallest to largest
	vid = mpy.VideoFileClip(video_filename)
	
	# Check that seconds argument does not excede total video duration
	logging.debug('Video duration: %s seconds', vid.duration) ## DEBUG
	if seconds[-1] > vid.duration:
		logging.error('The length (%s seconds) is shorter then the number of seconds that we want a frame from (%s seconds)!', vid.duration, seconds[-1])
		exit(1)
	
	# Extract first frame with timestamp higher then what is requested. 
	logging.debug('Video rotation: %s', vid.rotation) ## DEBUG
	vid = video_rotation(vid)
	for i, (tstamp, frame) in enumerate(vid.iter_frames(with_times=True)):
		if tstamp > seconds[0]:
			logging.info('Found frame for %s seconds: frame_count:%s; timestamp:%s', seconds[0], i, tstamp) ## DEBUG
			img = Image.fromarray(frame, 'RGB')
			frame_filename = out_prefix + '.' + str(seconds[0]) + 'sec.png'
			img.save(frame_filename)
			seconds = seconds[1:] # Remove first element from list as we just found a frame for this timepoint
		# Break loop if we have run out of timepoints that we want.
		if len(seconds) == 0:
			logging.info("Done extracting frames from video")
			break



# Rotate video back to 0
# Code based on https://github.com/Zulko/moviepy/issues/586
# 
# Parameters:
#	video	video file to process
def video_rotation(video):
	rotation = video.rotation
	if rotation == 0:
		video = video.rotate(-90)
	elif rotation == 90:
		video = video.resize(video.size[::-1])
		video.rotation = 0
	elif rotation == 180:
		video = video.rotate(90)
	elif rotation == 270:
		video = video.resize(video.size[::-1])
		video = video.rotate(180)  # Moviepy can only cope with 90, -90, and 180 degree turns
	else:
		logging.warning('Video has a weird rotation (i.e., not 0, 90, 180, or 270) of %s!', rotation)
	return video



# Extract mean value of RGB channels combined for a given image
# 
# Parameters:
#	image_filename	image file to get average RGB value for
def extract_colors(image_filename):
	pic = imageio.imread(image_filename)
	R = pic[ :, :, 0]
	G = pic[ :, :, 1]
	B = pic[ :, :, 2]
	meanR = np.mean(R)
	meanG = np.mean(G)
	meanB = np.mean(B)
	return ((meanR+meanG+meanB)/3)



