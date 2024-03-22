import sys
import os
import logging
import numpy as np
import copy as copy
from moviepy import *
import moviepy.editor as mpy
from PIL import Image, ImageDraw

from src.Utilities.Video_Results import Video_Results
from src.Utilities.constants import standards_color_space_values
from src.image import *

# To catch warnings from videos that are too short
import warnings

from src.utils import write_rgb_vals_to_csv, update_standard_deviation, adjust_color_space_values, \
    get_filename_from_path


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

    # Times to extract from video - make unique and sort.
    times = sorted(set([x[1] for x in test_analysis_times]))
    video_results = {}

    result_csv_file = videos[0] + outdir_suffix + '.result.csv'
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
        # Dictionary containing the predictions by the model for different time frames.
        # Key: time, value: prediction results from the model
        predictions_for_frames = {}
        hue_shifts = []
        shift = 0
        while shift < 360:
            hue_shifts.append(shift)
            shift += 30
        ## Crop tests from each time frame
        for time in times:
            frame_in = frame_prefix + "." + str(time) + "sec.png"
            frame_out = frame_prefix + "." + str(time) + "sec.detect"
            detection_images.append(frame_prefix + "." + str(time) + "sec.detect.detection.png")

            logging.info('Searching for tests in time %s seconds image', time)  ## INFO
            logging.debug('In frame: %s', frame_in)  ## DEBUG
            logging.debug('Out prefix: %s', frame_out)  ## DEBUG

            prediction = run_detector_on_image(frame_in, frame_out,
                                               model_detector_path, hue_shifts)
            predictions_for_frames[str(time)] = prediction

        # sys.exit(0)
        ## Open results file
        results = open(results_file, 'w')
        logging.info(f'Generating RGB values for the different categories at the specified time stamps')
        # TODO: should we calculate the RGB values of categories at all time frames
        # Generating RGB for each test crop from the specificed time frame.
        test_results_by_test_name = {}
        video_name = get_filename_from_path(video)
        logging.info(f'Hue shifts:{hue_shifts}')
        for hue_shift in hue_shifts:
            # Extracting the RGB values for the Standard colors.
            # We will be using these standard values to adjust the values for the predicted boxes and reduce the affect
            # of lightning
            for test_name, time in test_analysis_times:
                prediction_for_time_frame = predictions_for_frames[str(time)]
                standards = {'Red': 'Standard-Red', 'Green': 'Standard-Green', 'Blue': 'Standard-Blue'}
                deviation_from_standard = color_space_values()
                for key in standards:
                    target_frame = os.path.join(frame_prefix + "." + str(time) + "sec.detect.crop",
                                                standards[key] + ".png")
                    logging.debug('Searching for %s test in %s', standards[key], target_frame)  # DEBUG
                    color_values = extract_colors(target_frame)
                    update_standard_deviation(standards_color_space_values[key], color_values, deviation_from_standard)
                logging.debug(f'The deviation from standard RGB values for the time frame {time} seconds, '
                              f'are: {deviation_from_standard}')
                # Extract target crop and time
                target_frame = os.path.join(frame_prefix + "." + str(time) + "sec.detect.crop", test_name + ".png")
                logging.debug('Searching for %s test in %s', test_name, target_frame)  ## DEBUG
                test_color_space_values = extract_colors(target_frame)
                logging.debug('RGB: %s', test_color_space_values)  # DEBUG
                adj_test_color_space_values = adjust_color_space_values(test_color_space_values,
                                                                        deviation_from_standard)
                logging.debug('Color space values: %s', adj_test_color_space_values)  # DEBUG
                test_results_by_test_name[test_name] = adj_test_color_space_values
                results.write(test_name + '_hue_shift_' + str(hue_shift) + '_RGB_score\t' + str(
                    adj_test_color_space_values.rgb_score) + '\n')
                results.write(test_name + '_hue_shift_' + str(hue_shift) + '_Red_score\t' + str(
                    adj_test_color_space_values.red) + '\n')
                results.write(test_name + '_hue_shift_' + str(hue_shift) + '_Green_score\t' + str(
                    adj_test_color_space_values.green) + '\n')
                results.write(test_name + '_hue_shift_' + str(hue_shift) + '_Blue_score\t' + str(
                    adj_test_color_space_values.blue) + '\n')
                results.write(test_name + '_hue_shift_' + str(hue_shift) + '_Cyan_score\t' + str(
                    adj_test_color_space_values.cyan) + '\n')
                results.write(test_name + '_hue_shift_' + str(hue_shift) + '_Magenta_score\t' + str(
                    adj_test_color_space_values.magenta) + '\n')
                results.write(test_name + '_hue_shift_' + str(hue_shift) + '_Yellow_score\t' + str(
                    adj_test_color_space_values.yellow) + '\n')
                results.write(test_name + '_hue_shift_' + str(hue_shift) + '_Key_Black_score\t' + str(
                    adj_test_color_space_values.key_black) + '\n')
                results.write(test_name + '_hue_shift_' + str(hue_shift) + '_L_star_score\t' + str(
                    adj_test_color_space_values.l_star) + '\n')
                results.write(test_name + '_hue_shift_' + str(hue_shift) + '_a_star_score\t' + str(
                    adj_test_color_space_values.a_star) + '\n')
                results.write(test_name + '_hue_shift_' + str(hue_shift) + '_b_star_score\t' + str(
                    adj_test_color_space_values.b_star) + '\n')
            video_result = Video_Results()
            video_result.update_results_from_dictionary(test_results_by_test_name)
            if video_name not in video_results:
                logging.info(f'Creating new dictionary for the video:{video_name}')
                video_results[video_name] = {}
            logging.info(f'Video results for the video:{video_name} and hue shift:{hue_shift} is {video_result}')
            video_results[video_name]['_hue_shift_' + str(hue_shift)] = video_result
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
                                       append_images=detection_images_named[1:])
        ## Cleanup if required
        if cleanup:
            logging.info('Cleaning up - removing %s', outdir)  ## INFO
            shutil.rmtree(outdir)
        logging.info('# Finished. Results in %s', results_file)  ## INFO
    write_rgb_vals_to_csv(result_csv_file, video_results)
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
    times = sorted(set([x[1] for x in test_analysis_times]))

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
