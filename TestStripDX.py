#!/usr/bin/env python3
DESCRIPTION = '''
TestStripDX

An image processing framework for processing and extracting test strip results from a photo.

'''
import sys
import os
import argparse
from argparse import RawTextHelpFormatter
import logging
import shutil
from PIL import ImageDraw
from PyPDF2 import PdfFileMerger
from src.video import *
from src.detector import *


## Pass arguments.
def main():
	## Pass command line arguments. 
	parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=DESCRIPTION)
	subparsers = parser.add_subparsers(dest='command', required=True)
	
	## Parser for the conversion of the yolov4 to Tensorflow detector
	parser_convert_detector = subparsers.add_parser('convert', 
		help='Convert yolov4 detector to Tensorflow detector',
		description=CONVERT_RESULTS_DESCRIPTION
	)
	parser_convert_detector.add_argument('-m', '--model', metavar='model_name', 
		required=True, type=str, default='URS10', 
		help='Name of model in models/ directory to convert (NOTE: models/model_name.* file needs to exist)'
	)
	parser_convert_detector.add_argument('--debug',
		required=False, action='store_true',
		help='Print DEBUG info (default: %(default)s)'
	)
	
	## Parser for the processing of the test strip video files
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
	
	## Parser for the combining of the results files into a single output
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
	
	## Parser for command to join PDFs
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
	
	## Parse all arguments.
	args = parser.parse_args()
	
	## Set up basic debugger
	logFormat = "[%(levelname)s]: %(message)s"
	logging.basicConfig(format=logFormat, stream=sys.stderr, level=logging.INFO)
	if args.debug:
		logging.getLogger().setLevel(logging.DEBUG)
	
	logging.debug('%s', args) ## DEBUG
	
	## Model variables
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
	
	# Open targets file and convert to set(set(str, int), set(str, int), ...)
	model_targets = set()
	with open(model_targets_f, 'r') as fh:
		for line in fh:
			line = line.strip()
			# Ignore empty or commented out characters
			if line.startswith('#') or not line:
				continue
			
			x, y = line.split('\t')
			model_targets.add((x, int(y)))
	logging.debug('model_targets: %s', model_targets) ## DEBUG
	
	## Run subcommand
	if args.command == 'convert':
		convert_detector(model_weights, model_names, model_detector)
	elif args.command == 'process':
		process_videos(args.videos, model_detector, model_names, model_targets, args.cleanup, args.suffix)
	elif args.command == 'combine':
		combine_results(args.test_results, args.blank_results, args.out, model_targets)
	elif args.command == 'joinPDFs':
		joinPDFs(args.infiles, args.out)




CONVERT_RESULTS_DESCRIPTION = '''

Convert yolov4 detector to Tensorflow detector

Needs models/model_name.weights and models/model_name.names to exist.

'''
# Paramaters:
#	model_weights		path to weights file (e.g., 'models/teststrips.weights')
#	model_names		path to names file (e.g., 'models/teststrips.names')
#	output			path to output (e.g., 'models/teststrips.detector')
#	outdir_overwrite	overwrite output director directory (default: True)

def convert_detector(model_weights, model_names, output, outdir_overwrite=True):
	logging.info('Converting %s into Tensorflow model (output into %s)', model_weights, output) ## INFO
	
	## Check if weigths and names files exist
	EXIT = False
	if not os.path.exists(model_weights):
		EXIT = True
		logging.error('Weights file (%s) does not exist!', model_weights) ## ERROR
	if not os.path.exists(model_names):
		EXIT = True
		logging.error('Names file (%s) does not exist!', model_names) ## ERROR
	if EXIT:
		sys.exit(1)
	
	## Remove exisiting model directory if it exists
	if os.path.exists(output):
		if outdir_overwrite:
			logging.info('Detector directory %s already exists (from a previous run?) removing it so we can recreate it', output) ## INFO
			shutil.rmtree(output)
		else:
			logging.error('Detector directory %s already exists (from a previous run?), will not overwrite!') ## ERROR
			sys.exit(1)
	
	## Run detector converter function
	convert_to_Tensorflow_detector(model_weights, model_names, output)
	
	logging.info('Done converting model') ## INFO




PROCESS_VIDEOS_DESCRIPTION = '''

Runs Tensorflow model on each provided video on frames extracted at the specificed time points
for each test on the trip that we are interested in.

'''
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





COMBINE_RESULTS_DESCRIPTION = '''

Combine results from TestStripDX and calculate Relative Enzymatic Activity (REA) for each strip. 

If the optional blank results file is provided then REA values are calcultaed using those intensity values.
If a blank is not provided then default "blank" values of 255 will be used to calculate REA.

'''
def combine_results(test_files, blank_file, outfile, intervals):
	logging.info('Combining results files') ## INFO
	
	## Extract names and times to extract from video.
	names = [x[0] for x in intervals]
	
	## Header
	t = 'Files\ttype'
	for name in names:
		t = t + '\t' + name + '_intensity\t' + name + '_REA'
	outfile.write(t + '\n')

	## Blank strips
	logging.debug('Loading blank results file: %s', blank_file.name) ## DEBUG
	blank = load_results(blank_file, names)
	t = blank_file.name + '\tblank'
	for name in names:
		t = t + '\t' + str(blank[name]) + '\tNA'
	outfile.write(t + '\n')

	## Test strips
	for test_file in test_files:
		logging.debug('Loading results file: %s', test_file.name) ## DEBUG
		results = load_results(test_file, names)
		REA = {}
		for name in names:
			try:
				REA[name] = blank[name] - results[name]
			except TypeError:
				REA[name] = results[name]
		
		t = test_file.name + '\ttest'
		for name in names:
			t = t + '\t' + str(results[name]) + '\t' + str(REA[name])
		outfile.write(t + '\n')
	
	logging.info('Done combining results files') ## INFO



def load_results(results_file, names, delimiter='\t'):
	results = {x:255.0 for x in names}
	## results_file will be None if no blank file name is given. In these cases just use 255 as default "blank" values
	if results_file is not None:
		## Check that results file exists
		if not os.path.exists(results_file.name):
			logging.error('Results file (%s) does not exist!', results_file.name) ## ERROR
			sys.exit(1)
		
		## For each line
		with results_file as r:
			for line in r:
				line = line.strip()
				## Skip comment or blank lines
				if line.startswith('#') or not line:
					continue
				
				line_split = line.split(delimiter)
				if line_split[0] in names:
					## Will get a ValueError if we have a 'NA' missing value
					try:
						results[line_split[0]] = float(line_split[1])
					except ValueError:
						results[line_split[0]] = line_split[1]
		
		# Assume if results equals 255 assume that this name was missing from the results file.
		for name in names:
			if results[name] == 255.0:
				logging.warning('A value for %s was not found in %s results file. Setting to 255 as default.', name, results_file.name)
	logging.debug('%s', results) ## DEBUG
	return results



JOINPDFS_DESCRIPTION = '''

Takes a list of PDF files (either from command line or from stdin) and merges them into a single multipage document

NOTE:
        - Depending on the PDFs being merged this script might produce a few warnings:
                PdfReadWarning: Multiple definitions in dictionary at byte 0x1f0e for key /F3 [generic.py:588]
                PdfReadWarning: Multiple definitions in dictionary at byte 0x1f0e for key /F3 [generic.py:588]
                ...
          Nothing we can do to fix these problems (it has to do with how the PDFs are formed) so just ignore them. 

'''
def joinPDFs(input_PDFs, output_file):
	## See https://stackoverflow.com/questions/3444645/merge-pdf-files
	merger = PdfFileMerger(strict=False)
	
	for pdf in input_PDFs:
		merger.append(pdf)	
	
	merger.write(output_file)
	merger.close()



if __name__ == '__main__':
	main()
