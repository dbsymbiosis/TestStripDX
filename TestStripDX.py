#!/usr/bin/env python3
import sys
import os
import argparse
from argparse import RawTextHelpFormatter
import logging
import subprocess

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
                                  required=True, nargs='+', type=str,
                                  help='Video files to process'
                                  )
parser_process_video.add_argument('-m', '--model', metavar='model_name',
                                  required=False, type=str, default='URS10',
                                  help='Name of test strip being run. (default: %(default)s). Must have downloaded model files in models/ directory.'
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

##
## Parser for predicting frames of the test strip video files and upload these predictions to roboflow project
##
PROCESS_VIDEOS_UPLOAD_DESCRIPTION = '''
Runs Tensorflow model on each provided video on frames extracted at the specified time points
for each test on the strip that we are interested in. Later, the predicted bounding boxes along 
with the images are uploaded to roboflow.
'''
parser_process_video_and_roboflow = subparsers.add_parser('predict_and_upload_to_roboflow',
                                                          help='Process test strip video files and upload predicted images to roboflow',
                                                          description=PROCESS_VIDEOS_UPLOAD_DESCRIPTION)
parser_process_video_and_roboflow.add_argument('-i', '--in_videos', metavar='teststrip.mp4',
                                               required=True, nargs='+', type=str,
                                               help='Video files to process'
                                               )
parser_process_video_and_roboflow.add_argument('-m', '--model', metavar='model_name',
                                               required=False, type=str, default='URS10',
                                               help='Name of test strip being run. (default: %(default)s). Must have downloaded model files in models/ directory.'
                                               )
parser_process_video_and_roboflow.add_argument('-s', '--suffix', metavar='TestStripDX',
                                               required=False, type=str, default='.TestStripDX',
                                               help='Prefix to add to TestStripDX output files (default: %(default)s)'
                                               )
parser_process_video_and_roboflow.add_argument('-ak', '--apikey',
                                               required=True, type=str, default='',
                                               help='Api-key to access the roboflow project')
parser_process_video_and_roboflow.add_argument('-p', '--project',
                                               required=True, type=str, default='',
                                               help='Roboflow project name, where the annotations have to be uploaded to')
parser_process_video_and_roboflow.add_argument('-o', '--output_text_path', metavar='output_text_path',
                                               required=True, type=str,
                                               help='Path to the output text file, which the program can use to save bounding boxes data')
parser_process_video_and_roboflow.add_argument('--debug',
                                  required=False, action='store_true',
                                  help='Print DEBUG info (default: %(default)s)'
                                  )
##
## Parser for the combining of the results files into a single output
##
COMBINE_RESULTS_DESCRIPTION = '''

Combine results from TestStripDX and calculate Relative Enzymatic Activity (REA) for each strip. 

If the optional blank results file is provided then REA values are calcultaed using those intensity values.
If a blank is not provided then default "blank" values of 255 will be used to calculate REA.

'''
parser_combine_results = subparsers.add_parser('combine',
                                               help='Combine results from processed video files',
                                               description=COMBINE_RESULTS_DESCRIPTION
                                               )
parser_combine_results.add_argument('-i', '--in_results', metavar='test_results.txt',
                                    required=True, nargs='+', type=argparse.FileType('r'),
                                    help='Input test strip results files (required)'
                                    )
parser_combine_results.add_argument('-o', '--out_combined', metavar='combined.txt',
                                    required=False, default=sys.stdout, type=argparse.FileType('w'),
                                    help='Output file (default: stdout)'
                                    )
parser_combine_results.add_argument('-m', '--model', metavar='model_name',
                                    required=False, type=str, default='URS10',
                                    help='Name of test strip being run. (default: %(default)s). Must have downloaded model files in models/ directory.'
                                    )
parser_combine_results.add_argument('--debug',
                                    required=False, action='store_true',
                                    help='Print DEBUG info (default: %(default)s)'
                                    )

##
## Parser for command to join PDFs
##
JOINPDFS_DESCRIPTION = '''

Takes a list of PDF files (either from command line or from stdin) and merges them into a single multipage document.

NOTE:
        - Depending on the PDFs being merged this script might produce a few warnings:
                PdfReadWarning: Multiple definitions in dictionary at byte 0x1f0e for key /F3 [generic.py:588]
                PdfReadWarning: Multiple definitions in dictionary at byte 0x1f0e for key /F3 [generic.py:588]
                ...
          Nothing we can do to fix these problems (it has to do with how the PDFs are formed) so just ignore them. 

'''
parser_joinPDFs = subparsers.add_parser('joinPDFs',
                                        help='Join PDF files together',
                                        description=JOINPDFS_DESCRIPTION,
                                        formatter_class=RawTextHelpFormatter
                                        )
parser_joinPDFs.add_argument('-i', '--in_pdfs', nargs='+', metavar="file.pdf",
                             required=False, default=sys.stdin, type=str,
                             help='Input pdf files (default: stdin)'
                             )
parser_joinPDFs.add_argument('-o', '--out_pdf', metavar='combined.pdf',
                             required=True, type=str,
                             help='Output merged pdf file.'
                             )
parser_joinPDFs.add_argument('--debug',
                             required=False, action='store_true',
                             help='Print DEBUG info (default: %(default)s)'
                             )

##
## Parser for command to join PDFs
##
EXTRACT_FRAMES_DESCRIPTION = '''

Extract frames from a video at key time points.

This function is mostly used for collecting images to train the ML model.
'''
parser_extract_frames = subparsers.add_parser('extract',
                                              help='Extract frames from video',
                                              description=EXTRACT_FRAMES_DESCRIPTION,
                                              formatter_class=RawTextHelpFormatter
                                              )
parser_extract_frames.add_argument('-i', '--in_videos', metavar='teststrip.mp4',
                                   required=True, nargs='+', type=str,
                                   help='Video files to extract frames from.'
                                   )
parser_extract_frames.add_argument('-t', '--times', metavar=10,
                                   required=False, nargs='+', type=int, default=None,
                                   help='Time points at which to extract frames (default: use the times from --model).'
                                   )
parser_extract_frames.add_argument('-m', '--model', metavar='model_name',
                                   required=False, type=str, default='URS10',
                                   help='Name of test strip being run. (default: %(default)s). Must have downloaded model files in models/ directory.'
                                   )
parser_extract_frames.add_argument('-o', '--outdir', metavar='extracted_frames',
                                   required=False, type=str, default='extracted_frames',
                                   help='Directory where we will output the extracted frames (default: %(default)s)'
                                   )
parser_extract_frames.add_argument('--debug',
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

## Set envs for commands that use a model
# if args.command != 'joinPDFs':
script_dir = os.path.abspath(os.path.dirname(__file__))
models_dir = 'models'
model_params_path = os.path.join(script_dir, models_dir, args.model + '.py')
model_detector_path = os.path.join(script_dir, models_dir, args.model + '.pt')

## Model variables
## Import model params
# import model_params_path
from src.model_params import *

TEST_ANALYSIS_TIMES = get_test_analysis_times()
## Extract just the times from list of test names and times.
times = sorted(set([x[1] for x in TEST_ANALYSIS_TIMES]))

if args.command in ['process', 'combine']:
    ## Check model files exist
    logging.info('Checking model files (%s/%s.*) exist', models_dir, args.model)  ## INFO
    for file_path in [model_params_path, model_detector_path]:
        if not os.path.exists(file_path):
            logging.error('Model file (%s) does not exist!', file_path)  ## ERROR
            sys.exit(1)

    ## Import model params
    import model_params_path

elif args.command in ['extract']:
    times = []
    if args.times != None:
        times = args.times
    logging.debug('seconds: %s', times)  ## DEBUG


## Run subcommand
#	NOTE: Import each set of functions as needed becuase many of the packages take >30 sec to import
#	      so we need to only run import when we need to
if args.command == 'process':
    from src.video import *

    process_videos(args.in_videos,
                   model_detector_path,
                   times,  # Loaded from model_params_path
                   args.cleanup, args.suffix)
if args.command == 'predict_and_upload_to_roboflow':
    from src.video import *

    process_videos_and_upload_to_roboflow(args.in_videos, model_detector_path, times,
                                          args.suffix, args.apikey, args.project, args.output_text_path)
elif args.command == 'combine':
    from src.merge import *

    combine_results(args.in_results, args.out_combined, times)
elif args.command == 'joinPDFs':
    from src.merge import *

    joinPDFs(args.in_pdfs, args.out_pdf)
elif args.command == 'extract':
    from src.extract import *

    extract(args.in_videos, args.outdir, times)

logging.info('########################################################')  ## INFO
logging.info('                   TestStripDX Finished                 ')  ## INFO
logging.info('########################################################')  ## INFO
