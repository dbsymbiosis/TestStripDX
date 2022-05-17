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
import moviepy.editor as mpy
from PIL import Image

## Pass arguments.
def main():
	## Pass command line arguments. 
	parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=DESCRIPTION)
	parser.add_argument('-i', '--input_video', metavar='input.md4', 
		required=True, type=str, 
		help='Input video file (required)'
	)
	parser.add_argument('-o', '--out_frame', metavar='output.png', 
		required=True, type=str, 
		help='First frame from timestamp (required)'
	)
	parser.add_argument('-s', '--seconds', 
		required=True, type=int, 
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
	
	capture_frame(args.input_video, args.out_frame, args.seconds)



def capture_frame(video_filename, frame_filename, seconds):
	vid = mpy.VideoFileClip(video_filename)

	# Check that seconds argument does not excede total video duration
	logging.debug('Video duration: %s seconds', vid.duration) ## DEBUG
	if seconds > vid.duration:
		logging.error('The length (%s seconds) is shorter then the number of seconds that we want a frame from (%s seconds)!', vid.duration, seconds)
		exit(1)
	
	# Extract first frame with timestamp higher then what is requested. 
	for i, (tstamp, frame) in enumerate(vid.iter_frames(with_times=True)):
		if tstamp > seconds:
			logging.debug('i:%s; timestamp:%s', i, tstamp) ## DEBUG
			img = Image.fromarray(frame, 'RGB')
			img.save(frame_filename)
			break


if __name__ == '__main__':
	main()
