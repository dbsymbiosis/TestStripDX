#!/usr/bin/env python3
import sys
import os
import argparse
from argparse import RawTextHelpFormatter
import logging
import subprocess
import torch

def execute_commands():
    ## Get git hash and branch to use as program version
    cwd = os.path.dirname(os.path.realpath(__file__))
    git_branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=cwd).decode('ascii').strip()
    git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=cwd).decode('ascii').strip()
    __version__ = git_branch + ' ' + git_hash
    
    
    
    ##
    ## Pass command line arguments.
    ##
    DESCRIPTION = '''
    
    TestStripDX Version: {version}
    
    An image processing framework for processing and extracting test strip results from a photo.
    
    '''.format(version=__version__)
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=DESCRIPTION)
    parser.add_argument('-v', '--version', action='version', version=__version__)
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    
    
    ##
    ## Parser for the processing of the test strip video files
    ##
    PROCESS_VIDEOS_DESCRIPTION = '''
    
    Runs Tensorflow model on each provided video on frames extracted at the specificed time points
    for each test on the trip that we are interested in.
    
    '''
    parser_process_video = subparsers.add_parser('process',
                                                 help='Process test strip video files',
                                                 description=PROCESS_VIDEOS_DESCRIPTION)
    parser_process_video.add_argument('-i', '--in_videos', metavar='teststrip.mp4',
                                      required=False, nargs='+', type=str,
                                      help='Video files to process'
                                      )
    parser_process_video.add_argument('-ip', '--in_videos_dir',
                                      required=False, type=str,default='',
                                      help='Path to the directory containing the videos to be processed'
                                      )
    parser_process_video.add_argument('-m', '--model', metavar='model_name',
                                      required=False, type=str, default='URS10',
                                      help='Name of test strip being run. (default: %(default)s). '
                                           'Must have downloaded model files in models/ directory.'
                                      )
    parser_process_video.add_argument('-t', '--tests', metavar='tests', nargs='*',
                                      required=False, type=str, default=[],
                                      help='List of the tests to be run within the video. '
                                           'Provide the test names with spaces. Eg: -tests Glucose Blood'
                                      )
    parser_process_video.add_argument('-s', '--suffix', metavar='TestStripDX',
                                      required=False, type=str, default='.TestStripDX',
                                      help='Prefix to add to TestStripDX output files (default: %(default)s)'
                                      )
    parser_process_video.add_argument('-c', '--cleanup',
                                      required=False, action='store_true',
                                      help='Remove detection images from Tensorflow (default: %(default)s)'
                                      )
    parser_process_video.add_argument('--debug',
                                      required=False, action='store_true',
                                      help='Print DEBUG info (default: %(default)s)'
                                      )
    
    
    
    PROCESS_VIDEOS_DESCRIPTION = '''
    
    Runs Tensorflow model on each provided video on frames extracted at the specificed time points
    for each test on the trip that we are interested in.
    
    '''
    parser_process_video_2 = subparsers.add_parser('process_2',
                                                   help='Process test strip video files',
                                                   description=PROCESS_VIDEOS_DESCRIPTION)
    parser_process_video_2.add_argument('-i', '--in_videos', metavar='teststrip.mp4',
                                        required=False, nargs='+', type=str,
                                        help='Video files to process'
                                        )
    parser_process_video_2.add_argument('-ip', '--in_videos_dir',
                                        required=False, type=str,default='',
                                        help='Path to the directory containing the videos to be processed'
                                        )
    parser_process_video_2.add_argument('-m', '--model', metavar='model_name',
                                        required=False, type=str, default='URS10',
                                        help='Name of test strip being run. (default: %(default)s). '
                                             'Must have downloaded model files in models/ directory.'
                                        )
    parser_process_video_2.add_argument('-t', '--tests', metavar='tests', nargs='*',
                                        required=False, type=str, default=[],
                                        help='List of the tests to be run within the video. '
                                             'Provide the test names with spaces. Eg: -tests Glucose Blood'
                                        )
    parser_process_video_2.add_argument('-s', '--suffix', metavar='TestStripDX',
                                        required=False, type=str, default='.TestStripDX',
                                        help='Prefix to add to TestStripDX output files (default: %(default)s)'
                                        )
    parser_process_video_2.add_argument('-c', '--cleanup',
                                        required=False, action='store_true',
                                        help='Remove detection images from Tensorflow (default: %(default)s)'
                                        )
    parser_process_video_2.add_argument('--debug',
                                        required=False, action='store_true',
                                        help='Print DEBUG info (default: %(default)s)'
                                        )
    
    
    
    ##
    ## Parse all arguments.
    ##
    
    args = parser.parse_args()
    
    ## Set up basic debugger
    logFormat = "[%(levelname)s]: %(message)s"
    logging.basicConfig(format=logFormat, stream=sys.stderr, level=logging.INFO)
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logging.debug('%s', args)  ## DEBUG
    
    logging.info('########################################################')  ## INFO
    logging.info('                   TestStripDX Started                  ')  ## INFO
    logging.info('########################################################')  ## INFO
    logging.info('Version: ' + __version__)
    
    ##
    ## Set envs for commands that use a model
    ##
    # if args.command != 'joinPDFs' or 'train':
    script_dir = os.path.abspath(os.path.dirname(__file__))
    models_dir = 'models'
    model_detector_path = ''
    if args.command in ['train', 'process', 'combine', 'predict_and_upload_to_roboflow']:
        model_detector_path = os.path.join(script_dir, models_dir, args.model + '.pt')
        
        ## Check model files exist
        logging.info('Checking model files (%s/%s.*) exist', models_dir, args.model)  ## INFO
        for file_path in [model_detector_path]:
            if not os.path.exists(file_path):
                logging.error('Model file (%s) does not exist!', file_path)  ## ERROR
                sys.exit(1)
    
    
    
    ##
    ## Model variables
    ##
    # Import model params
    TEST_ANALYSIS_TIMES = []
    try:
        from src.common import get_test_analysis_times
        TEST_ANALYSIS_TIMES = get_test_analysis_times(args.tests)
    except:
        logging.error('No test arguments passed')
        # if args.command == #TODO: throw error when tests is required but not given(maybe make tests mandatory in the add_argument command)
    logging.debug(f'Test Analysis times: {TEST_ANALYSIS_TIMES}')  # DEBUG
    # Extract just the times from list of test names and times.
    times = sorted(set([x for x in TEST_ANALYSIS_TIMES]))
    
    if args.command in ['extract']:
        times = []
        if args.times != None:
            times = args.times
        logging.debug('seconds: %s', times)  ## DEBUG
    
    
    
    ##
    ## Run subcommand
    ##
    # NOTE: Import each set of functions as needed becuase many of the packages take >30 sec to import
    #       so we need to only run import when we need to
    if args.command == 'process':
        from src.utils import shift_hue
        for path in args.in_videos:
            predicted_class_names_paths = {'Standard-Blue':os.path.join(path, 'Standard-Blue.png'), 'Standard-Green':os.path.join(path, 'Standard-Green.png'), 'Standard-Red':os.path.join(path, 'Standard-Red.png'), 'Test-Blood':os.path.join(path, 'Test-Blood.jpg')}
            for predicted_class_name, cropped_img in predicted_class_names_paths.items():
                import cv2
                cropped_img = cv2.imread(cropped_img)
                cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
                for shift in [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]:
                    hue_shifted_img_dir = '.hueshifted' + str(shift)
                    hue_shifted_dir_path = os.path.join(path, hue_shifted_img_dir)
                    if not os.path.exists(hue_shifted_dir_path):
                        os.mkdir(hue_shifted_dir_path)
                    print(cropped_img)
                    hue_shifted_image = shift_hue(cropped_img, shift)
                    hue_shifted_img_name = predicted_class_name + '.png'
                    hue_shifted_image_path = os.path.join(hue_shifted_dir_path, hue_shifted_img_name)
                    cv2.imwrite(hue_shifted_image_path, hue_shifted_image)
    
    if args.command == 'process_2':
        import time
        import shutil
        from src.Utilities.color_space_values import color_space_values
        from src.Utilities.constants import standards_color_space_values
        from src.Utilities.Video_Results import Video_Results
        from src.image import extract_colors
        from src.utils import update_standard_deviation, adjust_color_space_values, write_rgb_vals_to_csv
    
        videos=args.in_videos
        model_detector_path=model_detector_path
        test_analysis_times=times
        cleanup=args.cleanup
        outdir_suffix=args.suffix
        outdir_overwrite=True
        in_vid_dir=args.in_videos_dir
        
        times = sorted(set([x[1] for x in test_analysis_times]))
        video_results = {}
        
        result_csv_file = videos[0] + outdir_suffix + '.result.csv'
        ## Process each video.
        for video in videos:
            starting_time = time.time()
            logging.info('# Extracting frames from %s', video)  ## INFO
            ## Envs
            outdir = video + outdir_suffix
            results_file = outdir + '.results.txt'
            frame_prefix = os.path.join(outdir, "frame")
            
            # Dictionary containing the predictions by the model for different time frames.
            # Key: time, value: prediction results from the model
            predictions_for_frames = {}
            hue_shifts = []
            shift = 0
            while shift < 360:
                hue_shifts.append(shift)
                shift += 30
            
            ## Open results file
            results = open(results_file, 'w')
            logging.info(f'Generating RGB values for the different categories at the specified time stamps')
            # TODO: should we calculate the RGB values of categories at all time frames
            # Generating RGB for each test crop from the specificed time frame.
            test_results_by_test_name = {}
            video_name = video.split('\\')[0]
            logging.info(f'Hue shifts:{hue_shifts}')
            for hue_shift in hue_shifts:
                # Extracting the RGB values for the Standard colors.
                # We will be using these standard values to adjust the values for the predicted boxes and reduce the affect
                # of lightning
                for test_name, frame_time in test_analysis_times:
                    frame_prefix_for_time_hue = os.path.join(frame_prefix + "." + str(frame_time) + "sec.detect.crop")
                    frame_prefix_for_time_hue = os.path.join(frame_prefix_for_time_hue, ".hueshifted"+str(hue_shift))
                    standards = {'Red': 'Standard-Red', 'Green': 'Standard-Green', 'Blue': 'Standard-Blue'}
                    deviation_from_standard = color_space_values()
                    for key in standards:
                        target_frame = os.path.join(frame_prefix_for_time_hue,
                                                    standards[key] + ".png")
                        logging.debug('Searching for %s test in %s', standards[key], target_frame)  # DEBUG
                        color_values,_ = extract_colors(target_frame)
                        update_standard_deviation(standards_color_space_values[key], color_values, deviation_from_standard)
                    logging.debug(f'The deviation from standard RGB values for the time frame {frame_time} seconds, '
                                  f'are: {deviation_from_standard}')
                    # Extract target crop and time
                    target_frame = os.path.join(frame_prefix_for_time_hue, test_name + ".png")
                    logging.debug('Searching for %s test in %s', test_name, target_frame)  ## DEBUG
                    test_color_space_values, is_prediction_available = extract_colors(target_frame)
                    logging.debug('RGB: %s', test_color_space_values)  # DEBUG
                    adj_test_color_space_values = test_color_space_values
                    if is_prediction_available:
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
            
            ## Cleanup if required
            if cleanup:
                logging.info('Cleaning up - removing %s', outdir)  ## INFO
                shutil.rmtree(outdir)
            logging.info('# Finished. Results in %s', results_file)  ## INFO
            logging.info(f'Finished processing video {video_name} in {time.time()-starting_time} seconds')
        write_rgb_vals_to_csv(result_csv_file, video_results)
        logging.info('####')  ## INFO
        logging.info('#### Finished processing video files')  ## INFO
        logging.info('####')  ## INFO
    
    
    
    logFormat = "[%(levelname)s]: %(message)s"
    logging.basicConfig(format=logFormat, stream=sys.stderr, level=logging.INFO)


if __name__ == '__main__':
    execute_commands()

