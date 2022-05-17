#!/usr/bin/env python
DESCRIPTION = '''
Extract frame from a specific timestamp in a video.
'''
import sys
import os
import argparse
import logging
import gzip
import sys
import numpy as np
from moviepy import *
import moviepy.editor as mpy
from PIL import Image

## Pass arguments.
def main():
	## Pass command line arguments. 
	parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=DESCRIPTION)
	parser.add_argument('-i', '--input_video', metavar='input.mp4', 
		required=True, type=str, 
		help='Input video file (required)'
	)
	parser.add_argument('-o', '--out_prefix', metavar='input',
		required=True, type=str, 
		help='Pefix to use for frames that we extract from video (required)'
	)
	parser.add_argument('-s', '--seconds', 
		required=True, type=int, nargs='+',
		help='Second into video to grab frame from (required)'
	)
	parser.add_argument('--debug', 
		required=False, action='store_true', 
		help='Print DEBUG info (default: %(default)s)'
	)
	args = parser.parse_args()
	
	## Set up basic debugger
	logFormat = "[%(levelname)s]: %(message)s"
	logging.basicConfig(format=logFormat, stream=sys.stderr, level=logging.INFO)
	if args.debug:
		logging.getLogger().setLevel(logging.DEBUG)
	
	logging.debug('%s', args) ## DEBUG
	
	capture_frame(args.input_video, args.out_prefix, args.seconds)



def capture_frame(video_filename, out_prefix, seconds):
	seconds = sorted(seconds) # sort so it is ordered smallest to largest
	vid = mpy.VideoFileClip(video_filename)

	# Check that seconds argument does not excede total video duration
	logging.debug('video rotation: %s', vid.rotation) ## DEBUG
	logging.debug('Video duration: %s seconds', vid.duration) ## DEBUG
	if seconds[-1] > vid.duration:
		logging.error('The length (%s seconds) is shorter then the number of seconds that we want a frame from (%s seconds)!', vid.duration, seconds[-1])
		exit(1)
	
	# Extract first frame with timestamp higher then what is requested. 
	vid = video_rotation(vid)
	for i, (tstamp, frame) in enumerate(vid.iter_frames(with_times=True)):
		if tstamp > seconds[0]:
			logging.debug('Found frame for %s seconds: frame_count:%s; timestamp:%s', seconds[0], i, tstamp) ## DEBUG
			img = Image.fromarray(frame, 'RGB')
			frame_filename = out_prefix + '.' + str(seconds[0]) + 'sec.png'
			img.save(frame_filename)
			seconds = seconds[1:] # Remove first element from list as we just found a frame for this timepoint
		# Break loop if we have run out of timepoints that we want.
		if len(seconds) == 0:
			break



def video_rotation(video):
	"""
	   Rotate video back to 0
	   Code based on https://github.com/Zulko/moviepy/issues/586
	"""
	if video.rotation in (90, 270):
		video = video.resize(video.size[::-1])
		video.rotation = 0
	else:
		video = video.rotate(90)
	return video



if __name__ == '__main__':
	main()
