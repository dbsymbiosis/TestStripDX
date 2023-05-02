import sys
import os
import logging
import numpy as np
from moviepy import *
import moviepy.editor as mpy
from PIL import Image, ImageDraw
from src.image import *

# To catch warnings from videos that are too short
import warnings
#warnings.filterwarnings('error')


# Process test strip video files
#
# Parameters:
#	videos			input video files
#	model_detector_path 	path to Tensorflow detector file (e.g., 'models/teststrips.detector')
#	model_names_path	path to names file (e.g., 'models/teststrips.names')
#	model_names		names of model objects
#	model_intervals		test intervals/coords
#	landmark_name		name of landmark to use for orientation
#	landmark_bounds		bounding coords of landmark
#	cleanup			cleanup temp files once finished processing video
#	outdir_suffix		suffix to add to output results files
#	outdir_overwrite	overwrite output director directory (default: True)
def process_videos(videos, 
		model_detector_path, model_names_path,
		model_names, model_intervals,
		landmark_name, landmark_bounds,
		cleanup, outdir_suffix, outdir_overwrite=True):
	logging.info('####') ## INFO
	logging.info('#### Processing video files') ## INFO
	logging.info('####') ## INFO
	
	## Times to extract from video - make unique and sort.
	# Add time 0 to list to use as blank
	times = [0] + sorted(set([x[1] for x in model_intervals]))
	
	for video in videos:
		logging.info('# Extracting frames from %s', video) ## INFO
		
		outdir = video+outdir_suffix
		results_file = outdir+'.results.txt'
		frame_prefix = os.path.join(outdir, "frame")
		detection_images = []
		detection_pdf_path = outdir+'.detection.pdf'
		
		## Check if video file exists.
		if not os.path.exists(video):
			logging.error('Video file %s does not exists!', video) ## ERROR
			sys.exit(1)
		
		## Remove exisiting model directory if it exists
		logging.debug('out_dir=%s', outdir) ## DEBUG
		if os.path.exists(outdir):
			if outdir_overwrite:
				logging.info('Output directory %s already exists (from a previous run?), removing it so we can recreate it', outdir) ## INFO
				shutil.rmtree(outdir)
			else:
				logging.error('Output directory %s already exists (from a previous run?), will not overwrite!', outdir) ## ERROR
				sys.exit(1)
		
		## Create output directory (after removing existing if present)
		os.mkdir(outdir)
		
		## Extract frame from a specific timestamp in a video.
		capture_frame(video, frame_prefix, times)
		
		## Crop tests from each time frame
		for time in times:
			frame_in = frame_prefix+"."+str(time)+"sec.png"
			frame_out = frame_prefix+"."+str(time)+"sec.detect"
			detection_images.append(frame_prefix+"."+str(time)+"sec.detect.detection.png")
			
			logging.info('Searching for tests in time %s seconds image', time) ## INFO
			logging.debug('In frame: %s', frame_in) ## DEBUG
			logging.debug('Out prefix: %s', frame_out) ## DEBUG
			
			crop_test_strip(frame_in, frame_out,
					model_detector_path, model_names_path, model_names, model_intervals,
					landmark_name, landmark_bounds)
		
		## Extract "blank" time 0 values for each test
		blank_values = {}
		for name, time, xmin, xmax, ymin, ymax in model_intervals:
			target_frame = os.path.join(frame_prefix+"."+str(0)+"sec.detect.crop", name+".png")
			logging.debug('Searching for %s test in %s', name, target_frame) ## DEBUG
			
			score = extract_colors(target_frame)
			logging.debug('Score: %s', score) ## DEBUG
			
			blank_values[name] = score
		
		## Open results file
		results = open(results_file, 'w')
		
		## Generate a score for each test crop from the specificed time frame.
		for name, time, xmin, xmax, ymin, ymax in model_intervals:
			target_frame = os.path.join(frame_prefix+"."+str(time)+"sec.detect.crop", name+".png")
			logging.debug('Searching for %s test in %s', name, target_frame) ## DEBUG
			
			score = extract_colors(target_frame)
			logging.debug('Score: %s', score) ## DEBUG
			
			adj_score = blank_values[name] - score
			logging.debug('Score: %s', adj_score) ## DEBUG
			
			results.write(name+'\t'+str(adj_score)+'\n')
		
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
		
		logging.info('# Finished. Results in %s', results_file) ## INFO
	
	logging.info('####') ## INFO
	logging.info('#### Finished processing video files') ## INFO
	logging.info('####') ## INFO



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
	
	# Extract first frame with timestamp higher then what is requested. 
	logging.debug('Video rotation: %s', vid.rotation) ## DEBUG
	vid = video_rotation(vid)
	last_valid_frame = []
	
	warnings.filterwarnings('error')
	try:
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
			
			# Save last valid frame incase we run out of video of the last times
			last_valid_frame = frame
	except Warning:
		logging.warning('Video is too short! Taking the last valid frame for times: %s', seconds) ## WARNING
		for time in seconds:
			img = Image.fromarray(last_valid_frame, 'RGB')
			frame_filename = out_prefix + '.' + str(time) + 'sec.png'
			img.save(frame_filename)
	warnings.filterwarnings('ignore')



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



