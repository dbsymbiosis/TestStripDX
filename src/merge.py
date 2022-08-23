import sys
import os
import logging
from PyPDF2 import PdfFileMerger



# Combine results from TestStripDX and calculate Relative Enzymatic Activity (REA) for each strip. 
#
#	test_files	results files from test videos
# 	blank_file	results file from blank video
#	outfile		output combined results file
#	intervals	time intervals to collect frames from videos
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
	if blank_file is not None:
		blank_name = blank_file.name
		logging.debug('Loading blank results file: %s', blank_name) ## DEBUG
	else:
		blank_name = 'NA'
		logging.debug('No blank file given. Using default blank values of 255 for REA calcualtion') ## DEBUG
	blank = load_results(blank_file, names)
	t = blank_name + '\tblank'
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
	## results_file will be None if no blank file name is given. In these cases just use 255 as default "blank" values
	if results_file is not None:
		results = {x:'NA' for x in names}
		
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
			if results[name] == 'NA':
				logging.warning('A value for %s was not found in %s results file.', name, results_file.name)
	else:
		results = {x:255 for x in names}
	logging.debug('%s', results) ## DEBUG
	return results



# Takes a list of PDF files (either from command line or from stdin) and merges them into a single multipage document
#
# 	input_PDFs	input PDF files
#	output_file	outout combined PDF file
def joinPDFs(input_PDFs, output_file):
	## See https://stackoverflow.com/questions/3444645/merge-pdf-files
	merger = PdfFileMerger(strict=False)
	
	for pdf in input_PDFs:
		merger.append(pdf)	
	
	merger.write(output_file)
	merger.close()



