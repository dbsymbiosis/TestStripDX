import sys
import os
import logging
import numpy as np
from moviepy import *
import moviepy.editor as mpy
from PIL import Image
from PIL import ImageDraw
import imageio.v2 as imageio
from src.detector import *



# Process test strip video files
#
# Parameters:
#	videos			input video files
#	model_detector		path to Tensorflow detector file (e.g., 'models/teststrips.detector')
#	model_names		path to names file (e.g., 'models/teststrips.names')
#	intervals		time intervals to collect frames from videos
#	cleanup			cleanup temp files once finished processing video
#	outdir_suffix		suffix to add to output results files
#	outdir_overwrite	overwrite output director directory (default: True)
def process_videos(videos, model_detector, model_names, intervals, cleanup, outdir_suffix, outdir_overwrite=True):
	logging.info('Processing video files') ## INFO
	
	## Check if detector directory exist
	EXIT = False
	if not os.path.exists(model_detector):
		EXIT = True
		logging.error('Detector (%s) does not exist! Try running the "convert" command first.', model_detector) ## ERROR
	if not os.path.exists(model_names):
		EXIT = True
		logging.error('Names file (%s) does not exist!', model_names) ## ERROR
	if EXIT:
		sys.exit(1)
	
	## Extract names and times to extract from video.
	names = [x[0] for x in intervals]
	times = [x[1] for x in intervals]
	
	for video in videos:
		outdir = video+outdir_suffix
		results_file = outdir+'.results.txt'
		frame_prefix = os.path.join(outdir, "frame")
		detection_images = []
		detection_pdf_path = outdir+'.detection.pdf'
		
		logging.info('########################################################') ## INFO
		logging.info('## Extracting frames from %s', video) ## INFO
		logging.info('########################################################') ## INFO
		
		## Create output directory
		logging.debug('out_dir=%s', outdir) ## DEBUG
		
		## Check if video file exists.
		if not os.path.exists(video):
			logging.error('Video file %s does not exists!', video) ## ERROR
			sys.exit(1)
		
		## Remove exisiting model directory if it exists
		if os.path.exists(outdir):
			if outdir_overwrite:
				logging.info('Output directory %s already exists (from a previous run?), removing it so we can recreate it', outdir) ## INFO
				shutil.rmtree(outdir)
			else:
				logging.error('Output directory %s already exists (from a previous run?), will not overwrite!', outdir) ## ERROR
				sys.exit(1)
		
		## Create output directory
		os.mkdir(outdir)
		
		## Open results file
		results = open(results_file, 'w')
		
		## Extract frame from a specific timestamp in a video.
		capture_frame(video, frame_prefix, times)
		
		## Extract tests from frames
		for name, time in intervals:
			frame_in = frame_prefix+"."+str(time)+"sec.png"
			frame_out = frame_prefix+"."+str(time)+"sec.detect"
			detection_images.append(frame_prefix+"."+str(time)+"sec.detect.detection1.png")
			
			logging.info('Searching %s frame at time %s seconds', name, time) ## INFO
			logging.debug('In frame: %s', frame_in) ## DEBUG
			logging.debug('Out prefix: %s', frame_out) ## DEBUG
			detect_test_strip(model_detector, model_names, [frame_in], frame_out, count=True, info=True)
			
			## Check to see if our target frame has been extracted
			target_frame = os.path.join(frame_prefix+"."+str(time)+"sec.detect.crop", name+"_1.png")
			logging.debug('Searching for cropped test: %s', target_frame) ## DEBUG
			if os.path.exists(target_frame):
				score = extract_colors(target_frame)
			else:
				logging.warning('No frame detected for %s frame at time %s seconds', name, time) ## WARNING
				score = 'NA'
			results.write(name+'\t'+str(score)+'\n')
			logging.debug('Score: %s', score) ## DEBUG
		
		## Close results file
		results.close()
		
		## Create combined detection image pdf
		logging.debug('detection_images: %s', detection_images) ## DEBUG
		detection_images_named = []
		for detection_image in detection_images:
			img = Image.open(detection_image)
			I1 = ImageDraw.Draw(img)
			I1.text((10, 30), detection_image, fill =(255, 0, 0))
			detection_images_named.append(img)
		detection_images_named[0].save(detection_pdf_path, 
			"PDF", resolution=1000.0, save_all=True, 
			append_images=detection_images_named[1:]
		)
		
		## Cleanup if required
		if cleanup:
			logging.info('Cleaning up - removing %s', outdir) ## INFO
			shutil.rmtree(outdir)
		
		logging.info('########################################################') ## INFO
		logging.info('## Finished. Results in %s', results_file) ## INFO
		logging.info('########################################################') ## INFO
	
	logging.info('Finished processing video files') ## INFO



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



