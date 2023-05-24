#!/usr/bin/env python3
import sys
import os
import argparse
from argparse import RawTextHelpFormatter
import logging
import subprocess


## Get git hash and branch to use as program version
cwd=os.path.dirname(os.path.realpath(__file__))
git_branch  = subprocess.check_output(['git', 'branch', '--show-current'], cwd=cwd).decode('ascii').strip()
git_hash    = subprocess.check_output(['git', 'rev-parse', 'HEAD'],        cwd=cwd).decode('ascii').strip()
__version__ = git_branch + ' ' + git_hash


##
## Pass command line arguments.
##
DESCRIPTION = '''

TestStripDX Version: {version}

An image processing framework for processing and extracting test strip results from a photo.

'''.format(version=__version__)
parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=DESCRIPTION)
subparsers = parser.add_subparsers(dest='command', required=True)

##
## Parser for the conversion of the yolov4 to Tensorflow detector
##
CONVERT_DETECTOR_DESCRIPTION = '''
Convert yolov4 detector to Tensorflow detector
Needs models/model_name.weights and models/model_name.names to exist.
'''
parser_convert_detector = subparsers.add_parser('convert', 
	help='Convert yolov4 detector to Tensorflow detector',
	description=CONVERT_DETECTOR_DESCRIPTION
)
parser_convert_detector.add_argument('-m', '--model', metavar='model_name',
	required=False, type=str, default='URS10',
	help='Name of test strip being run. (default: %(default)s). Must have downloaded model files in models/ directory.'
)
parser_convert_detector.add_argument('--debug',
	required=False, action='store_true',
	help='Print DEBUG info (default: %(default)s)'
)

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

Takes a list of PDF files (either from command line or from stdin) and merges them into a single multipage document

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
## Parse all arguments.
##
args = parser.parse_args()


## Set up basic debugger
logFormat = "[%(levelname)s]: %(message)s"
logging.basicConfig(format=logFormat, stream=sys.stderr, level=logging.INFO)
if args.debug:
	logging.getLogger().setLevel(logging.DEBUG)

logging.debug('%s', args) ## DEBUG


logging.info('########################################################') ## INFO
logging.info('                   TestStripDX Started                  ') ## INFO
logging.info('########################################################') ## INFO
logging.info('Version: ' + __version__)

## Model variables
if args.command != 'joinPDFs':
	script_dir = os.path.abspath(os.path.dirname(__file__))
	models_dir                 = 'models'
	model_weights_path         = os.path.join(script_dir, models_dir, args.model+'.weights')
	model_names_path           = os.path.join(script_dir, models_dir, args.model+'.names')
	model_intervals_path       = os.path.join(script_dir, models_dir, args.model+'.coords')
	model_landmark_bounds_path = os.path.join(script_dir, models_dir, args.model+'.landmark_bounds')
	model_light_standard_path  = os.path.join(script_dir, models_dir, args.model+'.light_standard')
	model_detector_path        = os.path.join(script_dir, models_dir, args.model+'.detector')
	
	## Check model files exist
	logging.info('Checking model files (%s/%s.*) exist', models_dir, args.model) ## INFO
	for file_path in [model_weights_path, model_names_path, model_intervals_path, model_landmark_bounds_path, model_intervals_path]:
		if not os.path.exists(file_path):
			logging.error('Model file (%s) does not exist!', file_path) ## ERROR
			sys.exit(1)
	if args.command != 'convert':
		if not os.path.exists(model_detector_path):
			logging.error('Detector file (%s) does not exist! Please run "convert" to create it from the weights file.', model_detector_path) ## ERROR
			sys.exit(1)
	
	## Open targets file and convert to [[str, int, float, float, float, float], [str, int, float, float, float, float], ...]
	logging.info('Opening file %s with list of test coods', model_intervals_path) ## INFO
	model_intervals = list()
	box_x = 40
	box_y = 40
	with open(model_intervals_path, 'r') as fh:
		for line in fh:
			line = line.strip()
			# Ignore empty or commented out characters
			if line.startswith('#') or not line:
				continue
			
			name, time, xmin, ymin = line.split('\t')
			model_intervals.append([name, int(time),
						float(xmin), float(xmin) + box_x,
						float(ymin), float(ymin) + box_y
						])
	logging.debug('model_intervals: %s', model_intervals) ## DEBUG
	
	## Open landmark name list file and convert to [str, str, ...]
	logging.info('Opening file %s with list of landmark objects to use for image orientation', model_names_path) ## INFO
	model_names = list()
	with open(model_names_path, 'r') as fh:
		for line in fh:
			line = line.strip()
			# Ignore empty or commented out characters
			if line.startswith('#') or not line:
				continue
			
			name = line.split('\t')[0]
			model_names.append(name)
	logging.debug('model_names: %s', model_names) ## DEBUG
	
	## Open file listing landmark bounds and convert to {"name":str, "xmin":int, "xmax":int, "ymin":int, "ymax":int}
	logging.info('Opening file %s with list of landmark objects to use for image orientation', model_names_path) ## INFO
	model_landmark_bounds = dict()
	with open(model_landmark_bounds_path, 'r') as fh:
		for line in fh:
			line = line.strip()
			# Ignore empty or commented out characters
			if line.startswith('#') or not line:
				continue
			
			line_split = line.split('\t')
			model_landmark_bounds["name"] = line_split[0]
			model_landmark_bounds["xmin"] = int(line_split[1])
			model_landmark_bounds["xmax"] = int(line_split[2])
			model_landmark_bounds["ymin"] = int(line_split[3])
			model_landmark_bounds["ymax"] = int(line_split[4])
			break # Only interested in the first non-blank/comment line
	logging.debug('model_landmark_bounds: %s', model_landmark_bounds) ## DEBUG
	
	## Open file listing position of white space to use as light standard and convert to {"name":str, "xmin":int, "xmax":int, "ymin":int, "ymax":int}
	logging.info('Opening file %s with list of landmark objects to use for image orientation', model_names_path) ## INFO
	model_light_standard = dict()
	with open(model_light_standard_path, 'r') as fh:
		for line in fh:
			line = line.strip()
			# Ignore empty or commented out characters
			if line.startswith('#') or not line:
				continue
			
			line_split = line.split('\t')
			model_light_standard["name"] = line_split[0]
			model_light_standard["xmin"] = int(line_split[1])
			model_light_standard["xmax"] = int(line_split[2])
			model_light_standard["ymin"] = int(line_split[3])
			model_light_standard["ymax"] = int(line_split[4])
			break # Only interested in the first non-blank/comment line
	logging.debug('model_light_standard: %s', model_light_standard) ## DEBUG


## Run subcommand
#	NOTE: Import each set of functions as needed becuase many of the packages take >30 sec to import
#	      so we need to only run import when we need to
if args.command == 'convert':
	from src.detector import *
	convert_detector(model_weights_path, model_names_path, model_detector_path)
elif args.command == 'process':
	from src.video import *	
	process_videos(args.in_videos,
			model_detector_path, model_names_path,
			model_names, model_intervals,
			model_landmark_bounds,
			model_light_standard,
			args.cleanup, args.suffix)
elif args.command == 'combine':
	from src.merge import *
	combine_results(args.in_results, args.out_combined, model_intervals)
elif args.command == 'joinPDFs':
	from src.merge import *
	joinPDFs(args.in_pdfs, args.out_pdf)



logging.info('########################################################') ## INFO
logging.info('                   TestStripDX Finished                 ') ## INFO
logging.info('########################################################') ## INFO



