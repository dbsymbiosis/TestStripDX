import sys
import os
import logging
import numpy as np
import copy as copy
from moviepy import *
import moviepy.editor as mpy
from PIL import Image, ImageDraw
from src.video import *

# To catch warnings from videos that are too short
import warnings
#warnings.filterwarnings('error')


# Extract frames from provided video files.
#
# Parameters:
#	videos		input video files
#	outdir		output frame directory
#	seconds		times in seconds to extract frames from each video
def extract(videos, outdir, seconds):
	## Times to extract from video - make unique and sort.
	seconds = sorted(set(seconds))
	
	## Create output dir
	os.makedirs(outdir, exist_ok=True)
	
	## Process each video
	for video in videos:
		logging.info('# Extracting frames from %s', video) ## INFO
		video_basename = os.path.basename(video)
		out_prefix = os.path.join(outdir, video_basename + '.frame')
		capture_frames_from_video(video, out_prefix, seconds)
		logging.info('# Done extracting frames from %s', video) ## INFO
	
	logging.info('####') ## INFO
	logging.info('#### Finished extracting frames from video files') ## INFO
	logging.info('####') ## INFO


