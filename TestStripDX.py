#!/usr/bin/env python3
import sys
import os
import argparse
from argparse import RawTextHelpFormatter
import logging



##
## Pass command line arguments.
##
DESCRIPTION = '''

TestStripDX

An image processing framework for processing and extracting test strip results from a photo.

'''
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
	required=True, type=str, default='URS10', 
	help='Name of model in models/ directory to convert (NOTE: models/model_name.* file needs to exist)'
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
parser_process_video.add_argument('-v', '--videos', metavar='teststrip.mp4', 
	required=True, nargs='+', type=str, 
	help='Video files to process'
)
parser_process_video.add_argument('-m', '--model', metavar='model_name',
	required=False, type=str, default='URS10',
	help='Name of model in models/ directory to convert (NOTE: models/model_name.* file needs to exist)'
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
parser_combine_results.add_argument('-t', '--test_results', metavar='test_results.txt',
	required=True, nargs='+', type=argparse.FileType('r'),
	help='Input test strip results files (required)'
)
parser_combine_results.add_argument('-o', '--out', metavar='output.txt',
	required=False, default=sys.stdout, type=argparse.FileType('w'),
	help='Output file (default: stdout)'
)
parser_combine_results.add_argument('-b', '--blank_results', metavar='blank_results.txt',
	required=False, default=None, type=argparse.FileType('r'),
	help='Input blank strip results files (default: not used)'
)
parser_combine_results.add_argument('-m', '--model', metavar='model_name',
	required=False, type=str, default='URS10',
	help='Name of model in models/ directory to convert (NOTE: models/model_name.* file needs to exist)'
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
parser_joinPDFs.add_argument('-i', '--infiles', nargs='+', metavar="file.pdf",
	required=False, default=sys.stdin, type=str,
	help='Input pdf files (default: stdin)'
)
parser_joinPDFs.add_argument('-o', '--out', metavar='merged.pdf',
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



## Model variables
if args.command != 'joinPDFs':
	script_dir = os.path.abspath(os.path.dirname(__file__))
	models_dir       = 'models'
	model_weights    = os.path.join(script_dir, models_dir, args.model+'.weights')
	model_names      = os.path.join(script_dir, models_dir, args.model+'.names')
	model_detector   = os.path.join(script_dir, models_dir, args.model+'.detector')
	model_targets_f  = os.path.join(script_dir, models_dir, args.model+'.targets')
	
	## Load targets info
	logging.info('Opening file %s with list of target tests', model_targets_f) ## INFO
	if not os.path.exists(model_targets_f):
		logging.error('Targets file (%s) does not exist!', model_targets_f) ## ERROR
		sys.exit(1)
	
	## Open targets file and convert to [[str, int], [str, int], ...]
	model_targets = list()
	with open(model_targets_f, 'r') as fh:
		for line in fh:
			line = line.strip()
			# Ignore empty or commented out characters
			if line.startswith('#') or not line:
				continue
			
			x, y = line.split('\t')
			model_targets.append([x, int(y)])
	logging.debug('model_targets: %s', model_targets) ## DEBUG



## Run subcommand
#	NOTE: Import each set of functions as needed becuase many of the packages take >30 sec to import
#	      so we need to only run import when we need to
if args.command == 'convert':
	from src.detector import *
	convert_detector(model_weights, model_names, model_detector)
elif args.command == 'process':
	from src.video import *
	process_videos(args.videos, model_detector, model_names, model_targets, args.cleanup, args.suffix)
elif args.command == 'combine':
	from src.merge import *
	combine_results(args.test_results, args.blank_results, args.out, model_targets)
elif args.command == 'joinPDFs':
	from src.merge import *
	joinPDFs(args.infiles, args.out)



