import sys
import os
import logging
import numpy as np
import copy as copy
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
#	videos				input video files
#	model_detector_path	 	path to Tensorflow detector file (e.g., 'models/teststrips.detector')
#	model_names_path		path to names file (e.g., 'models/teststrips.names')
#	model_names			names of model objects
#	model_intervals			test intervals/coords
#	model_landmark_bounds		bounding coords of landmark
#	model_color_standard_bounds	bounds of color standards (for light normalization)
#	cleanup				cleanup temp files once finished processing video
#	outdir_suffix			suffix to add to output results files
#	outdir_overwrite		overwrite output director directory (default: True)
def process_videos(videos, 
		model_detector_path, model_names_path,
		model_names, model_intervals,
		model_landmark_bounds,
		model_color_standard_bounds,
		cleanup, outdir_suffix, outdir_overwrite=True):
	logging.info('####') ## INFO
	logging.info('#### Processing video files') ## INFO
	logging.info('####') ## INFO
	
	## Add color standard(s) to list of interval to crop.
	model_intervals_tocrop = copy.deepcopy(model_intervals)
	model_intervals_tocrop.append([
					model_color_standard_bounds['name'], # name
					0,                                   # time
					model_color_standard_bounds['xmin'], # xmin
					model_color_standard_bounds['xmax'], # xmax
					model_color_standard_bounds['ymin'], # ymin
					model_color_standard_bounds['ymax'], # ymax
				])
	
	## Times to extract from video - make unique and sort.
	times = sorted(set([x[1] for x in model_intervals]))
	
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
		capture_frames(video, model_detector_path, model_names_path, model_names, model_landmark_bounds, frame_prefix, times)
		
		## Crop tests from each time frame
		for time in times:
			frame_in = frame_prefix+"."+str(time)+"sec.png"
			frame_out = frame_prefix+"."+str(time)+"sec.detect"
			detection_images.append(frame_prefix+"."+str(time)+"sec.detect.detection.png")
			
			logging.info('Searching for tests in time %s seconds image', time) ## INFO
			logging.debug('In frame: %s', frame_in) ## DEBUG
			logging.debug('Out prefix: %s', frame_out) ## DEBUG
			
			crop_test_strip(frame_in, frame_out,
					model_detector_path, model_names_path, model_names, 
					model_intervals_tocrop,
					model_landmark_bounds)
		
		## Open results file
		results = open(results_file, 'w')
		
		## Generate RGB for each test crop from the specificed time frame.
		for name, time, xmin, xmax, ymin, ymax in model_intervals:
			## Extract "blank" crop to use for light standardization
			target_frame = os.path.join(frame_prefix+"."+str(time)+"sec.detect.crop", model_color_standard_bounds['name']+".png")
			logging.debug('Searching for %s test in %s', model_color_standard_bounds['name'], target_frame) ## DEBUG
			
			RGB = extract_colors(target_frame)
			logging.debug('white standard RGB: %s', RGB) ## DEBUG
			
			blank_RGB = {}
			blank_RGB['score'] = 255 - RGB['score']
			blank_RGB['R']     = 255 - RGB['R']
			blank_RGB['G']     = 255 - RGB['G']
			blank_RGB['B']     = 255 - RGB['B']
			
			# Extract target crop and time
			target_frame = os.path.join(frame_prefix+"."+str(time)+"sec.detect.crop", name+".png")
			logging.debug('Searching for %s test in %s', name, target_frame) ## DEBUG
			
			RGB = extract_colors(target_frame)
			logging.debug('RGB: %s', RGB) ## DEBUG
			
			adj_RGB = {}
			adj_RGB['score'] = RGB['score'] + blank_RGB['score'] 
			adj_RGB['R']     = RGB['R']     + blank_RGB['R'] 
			adj_RGB['G']     = RGB['G']     + blank_RGB['G'] 
			adj_RGB['B']     = RGB['B']     + blank_RGB['B'] 
			logging.debug('RGB: %s', adj_RGB) ## DEBUG
			
			results.write(name+'_score\t'+str(adj_RGB['score'])+'\n')
			results.write(name+'_R\t'+str(adj_RGB['R'])+'\n')
			results.write(name+'_G\t'+str(adj_RGB['G'])+'\n')
			results.write(name+'_B\t'+str(adj_RGB['B'])+'\n')
		
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
#	video_filename		input video files
#	model_detector_path	path to Tensorflow detector file (e.g., 'models/teststrips.detector')
#	model_names_path	path to names file (e.g., 'models/teststrips.names')
#	model_names		names of model objects
#	model_landmark_bounde	dict of landmark features to check for in image
#	out_prefix		prefix to use for frames that we extract from video
#	seconds			second into video to grab frame from
def capture_frames(video_filename, model_detector_path, model_names_path, model_names, model_landmark_bounds, out_prefix, seconds):
	seconds = sorted(seconds) # sort so it is ordered smallest to largest
	vid = mpy.VideoFileClip(video_filename)
	logging.debug('Video duration: %s seconds', vid.duration) ## DEBUG
	logging.debug('Reported video rotation: %s', vid.rotation) ## DEBUG
	
	# Extract first frame with timestamp higher then what is requested. 
	logging.info('Checking and correcting video rotation') ## INFO
	vid = video_rotation(vid, model_detector_path, model_names_path, model_names, model_landmark_bounds, out_prefix)
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
#	video			video file to process
#	model_detector_path	path to Tensorflow detector file (e.g., 'models/teststrips.detector')
#	model_names_path	path to names file (e.g., 'models/teststrips.names')
#	model_names		names of model objects
#	model_landmark_bounde	dict of landmark features to check for in image
#	out_prefix		prefix to use for frames that we extract from video
#	test_frame_num		number of frame to use for rotation testing
def video_rotation(video, model_detector_path, model_names_path, model_names, model_landmark_bounds, out_prefix, test_frame_num=10):
	
	# Rotate 0
	logging.debug('Trying roation 0') ## DEBUG
	t_video = video.rotate(-90)
	if extract_and_check_rotation_frame(t_video, model_detector_path, model_names_path, model_names, model_landmark_bounds, out_prefix+'.rotate0', test_frame_num):
		logging.debug('- Success!') ## DEBUG
		return(t_video)
	
	# Rotate forward 90
	logging.debug('Trying roation 90') ## DEBUG
	t_video = video.resize(video.size[::-1])
	t_video.rotation = 0
	if extract_and_check_rotation_frame(t_video, model_detector_path, model_names_path, model_names, model_landmark_bounds, out_prefix+'.rotate90', test_frame_num):
		logging.debug('- Success!') ## DEBUG
		return(t_video)
	
	# Rotate forward 180
	logging.debug('Trying roation 180') ## DEBUG
	t_video = video.rotate(90)
	if extract_and_check_rotation_frame(t_video, model_detector_path, model_names_path, model_names, model_landmark_bounds, out_prefix+'.rotate180', test_frame_num):
		logging.debug('- Success!') ## DEBUG
		return(t_video)
	
	# Rotate forward 270:
	logging.debug('Trying roation 270') ## DEBUG
	t_video = video.resize(video.size[::-1])
	t_video = t_video.rotate(180)  # Moviepy can only cope with 90, -90, and 180 degree turns
	if extract_and_check_rotation_frame(t_video, model_detector_path, model_names_path, model_names, model_landmark_bounds, out_prefix+'.rotate270', test_frame_num):
		logging.debug('- Success!') ## DEBUG
		return(t_video)
	
	# If we havent found the correct rotation yet (or we could not find the landmark)
	logging.error('Video appears to have either a weird rotation (i.e., not 0, 90, 180, or 270; video says it has a rotation of %s) or we could not find the landmark (called %s). We will move ahead using the default roation but this will likely fail.', 
		video.rotation, model_landmark_bounds["name"]) ## ERROR
	return(video)


def extract_and_check_rotation_frame(video, model_detector_path, model_names_path, model_names, model_landmark_bounds, out_prefix, test_frame_num=10):
	frame_filename      = out_prefix+'.png'
	ML_frame_fileprefix = out_prefix
	warnings.filterwarnings('error')
	try:
		for i, (tstamp, frame) in enumerate(video.iter_frames(with_times=True)):
			if i == test_frame_num:
				img = Image.fromarray(frame, 'RGB')
				img.save(frame_filename)
				break
			# Save last valid frame incase we run out of video of the last times
			last_valid_frame = frame
	except Warning:
		logging.warning('Video is too short to identify rotation! Trying to take the last valid frame to fix this') ## WARNING
		img = Image.fromarray(last_valid_frame, 'RGB')
		img.save(frame_filename)
	warnings.filterwarnings('ignore')
	
	frame = cv2.imread(frame_filename)
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	landmark_found, l_xmin, l_ymin, l_xmax, l_ymax = check_landmark(frame,
		model_detector_path, model_names_path, model_names, model_landmark_bounds,
		ML_frame_fileprefix)
	return(landmark_found)



