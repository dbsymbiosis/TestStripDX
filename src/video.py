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


# warnings.filterwarnings('error')


# Process test strip video files
#
# Parameters:
#	videos				input video files
#	model_detector_path	 	path to detector file (e.g., 'models/teststrips.pt')
#	test_analysis_times		list of test names and times [ ['t1', 5], ['t2', 10], ..., ... ]
#	cleanup				cleanup temp files once finished processing video
#	outdir_suffix			suffix to add to output results files
#	outdir_overwrite		overwrite output director directory (default: True)
def process_videos(videos,
                   model_detector_path,
                   test_analysis_times,
                   cleanup, outdir_suffix, outdir_overwrite=True):
    logging.info('####')  ## INFO
    logging.info('#### Processing video files')  ## INFO
    logging.info('####')  ## INFO

    ## Times to extract from video - make unique and sort.
    times = sorted(set([x for x in test_analysis_times]))

    ## Process each video.
    for video in videos:
        logging.info('# Extracting frames from %s', video)  ## INFO

        ## Envs
        outdir = video + outdir_suffix
        results_file = outdir + '.results.txt'
        frame_prefix = os.path.join(outdir, "frame")
        detection_images = []
        detection_pdf_path = outdir + '.detection.pdf'
        ## Check if video file exists.
        if not os.path.exists(video):
            logging.error('Video file %s does not exists!', video)  ## ERROR
            sys.exit(1)

        ## Remove exisiting model directory if it exists
        logging.debug('out_dir=%s', outdir)  ## DEBUG
        if os.path.exists(outdir):
            if outdir_overwrite:
                logging.info(
                    'Output directory %s already exists (from a previous run?), removing it so we can recreate it',
                    outdir)  ## INFO
                shutil.rmtree(outdir)
            else:
                logging.error('Output directory %s already exists (from a previous run?), will not overwrite!',
                              outdir)  ## ERROR
                sys.exit(1)

        ## Create output directory (after removing existing if present)
        os.mkdir(outdir)

        ## Extract frame from a specific timestamp in a video.
        capture_frames_from_video(video, frame_prefix, times)

        ## Crop tests from each time frame
        for time in times:
            frame_in = frame_prefix + "." + str(time) + "sec.png"
            frame_out = frame_prefix + "." + str(time) + "sec.detect"
            detection_images.append(frame_prefix + "." + str(time) + "sec.detect.detection.png")

            logging.info('Searching for tests in time %s seconds image', time)  ## INFO
            logging.debug('In frame: %s', frame_in)  ## DEBUG
            logging.debug('Out prefix: %s', frame_out)  ## DEBUG

            run_detector_on_image(frame_in, frame_out,
                                  model_detector_path)

        sys.exit(0)
        ## Open results file
        results = open(results_file, 'w')

        ## Generate RGB for each test crop from the specificed time frame.
        for name, time, xmin, xmax, ymin, ymax in times:
            ## Extract "blank" crop to use for light standardization
            target_frame = os.path.join(frame_prefix + "." + str(time) + "sec.detect.crop",
                                        model_color_standard_bounds['name'] + ".png")
            logging.debug('Searching for %s test in %s', model_color_standard_bounds['name'], target_frame)  ## DEBUG

            RGB = extract_colors(target_frame)
            logging.debug('white standard RGB: %s', RGB)  ## DEBUG

            blank_RGB = {}
            blank_RGB['score'] = 255 - RGB['score']
            blank_RGB['R'] = 255 - RGB['R']
            blank_RGB['G'] = 255 - RGB['G']
            blank_RGB['B'] = 255 - RGB['B']

            # Extract target crop and time
            target_frame = os.path.join(frame_prefix + "." + str(time) + "sec.detect.crop", name + ".png")
            logging.debug('Searching for %s test in %s', name, target_frame)  ## DEBUG

            RGB = extract_colors(target_frame)
            logging.debug('RGB: %s', RGB)  ## DEBUG

            adj_RGB = {}
            adj_RGB['score'] = RGB['score'] + blank_RGB['score']
            adj_RGB['R'] = RGB['R'] + blank_RGB['R']
            adj_RGB['G'] = RGB['G'] + blank_RGB['G']
            adj_RGB['B'] = RGB['B'] + blank_RGB['B']
            logging.debug('RGB: %s', adj_RGB)  ## DEBUG

            results.write(name + '_score\t' + str(adj_RGB['score']) + '\n')
            results.write(name + '_R\t' + str(adj_RGB['R']) + '\n')
            results.write(name + '_G\t' + str(adj_RGB['G']) + '\n')
            results.write(name + '_B\t' + str(adj_RGB['B']) + '\n')

        ## Close results file
        results.close()

        ## Create combined detection image pdf
        logging.debug('detection_images: %s', detection_images)  ## DEBUG
        detection_images_named = []
        for detection_image in detection_images:
            img = Image.open(detection_image)
            I1 = ImageDraw.Draw(img)
            I1.text((10, 30), detection_image, fill=(255, 0, 0))
            detection_images_named.append(img)
        detection_images_named[0].save(detection_pdf_path,
                                       "PDF", resolution=1000.0, save_all=True,
                                       append_images=detection_images_named[1:]
                                       )

        ## Cleanup if required
        if cleanup:
            logging.info('Cleaning up - removing %s', outdir)  ## INFO
            shutil.rmtree(outdir)

        logging.info('# Finished. Results in %s', results_file)  ## INFO

    logging.info('####')  ## INFO
    logging.info('#### Finished processing video files')  ## INFO
    logging.info('####')  ## INFO


# Extract a frames from a given time point in the provided video file
#
# Parameters:
#	video_filename		input video files
#	out_prefix		prefix to use for frames that we extract from video
#	seconds			second into video to grab frame from
def capture_frames_from_video(video_filename, out_prefix, seconds):
    seconds = sorted(set(seconds))  # sort so it is ordered smallest to largest
    vid = mpy.VideoFileClip(video_filename)
    logging.debug('Video duration: %s seconds', vid.duration)  ## DEBUG

    ## Extract first frame with timestamp higher then what is requested.
    last_valid_frame = []
    warnings.filterwarnings('error')
    try:
        for i, (tstamp, frame) in enumerate(vid.iter_frames(with_times=True)):
            if tstamp > seconds[0]:
                logging.info('Found frame for %s seconds: frame_count:%s; timestamp:%s', seconds[0], i,
                             tstamp)  ## DEBUG
                img = Image.fromarray(frame, 'RGB')
                frame_filename = out_prefix + '.' + str(seconds[0]) + 'sec.png'
                img.save(frame_filename)
                seconds = seconds[1:]  # Remove first element from list as we just found a frame for this timepoint

            # Break loop if we have run out of timepoints that we want.
            if len(seconds) == 0:
                logging.info("Done extracting frames from video")
                break

            # Save last valid frame incase we run out of video of the last times
            last_valid_frame = frame
    except Warning:
        logging.warning('Video is too short! Taking the last valid frame for times: %s', seconds)  ## WARNING
        for time in seconds:
            img = Image.fromarray(last_valid_frame, 'RGB')
            frame_filename = out_prefix + '.' + str(time) + 'sec.png'
            img.save(frame_filename)
    warnings.filterwarnings('ignore')


def process_videos_and_upload_to_roboflow(videos, model_detector_path, test_analysis_times, outdir_suffix,
                                          api_key, project_name, output_txt_file_path):
    logging.info('####')  ## INFO
    logging.info('#### Processing video files')  ## INFO
    logging.info('####')  ## INFO

    ## Times to extract from video - make unique and sort.
    times = sorted(set([x for x in test_analysis_times]))

    ## Process each video.
    for video in videos:
        logging.info('# Extracting frames from %s', video)  ## INFO

        ## Envs
        outdir = video + outdir_suffix
        os.mkdir(outdir)
        frame_prefix = os.path.join(outdir, "frame")

        ## Check if video file exists.
        if not os.path.exists(video):
            logging.error('Video file %s does not exists!', video)  ## ERROR
            sys.exit(1)

        ## Extract frame from a specific timestamp in a video.
        capture_frames_from_video(video, frame_prefix, times)

        ## Crop tests from each time frame
        for time in times:
            frame_in = frame_prefix + "." + str(time) + "sec.png"
            logging.info('Searching for tests in time %s seconds image', time)  ## INFO
            logging.debug('In frame: %s', frame_in)  ## DEBUG

            predict_image_save_boxes(frame_in, model_detector_path, output_txt_file_path, api_key, project_name)
