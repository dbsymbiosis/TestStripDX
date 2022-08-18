#!/usr/bin/env python
DESCRIPTION = '''
Combine results from TestStripDX and calculate Relative Enzymatic Activity (REA) for each strip. 

If the optional blank results file is provided then REA values are calcultaed using those intensity values.
If a blank is not provided then default "blank" values of 255 will be used to calculate REA.
'''
import sys
import os
import argparse
import logging
import gzip

## Pass arguments.
def main():
	## Pass command line arguments. 
	parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=DESCRIPTION)
	parser.add_argument('-t', '--test_results', metavar='test_results.txt', 
		required=True, nargs='+', type=argparse.FileType('r'), 
		help='Input [gzip] test strip results files (required)'
	)
	parser.add_argument('-o', '--out', metavar='output.txt', 
		required=False, default=sys.stdout, type=argparse.FileType('w'), 
		help='Output [gzip] file (default: stdout)'
	)
	parser.add_argument('-b', '--blank_results', metavar='blank_results.txt', 
		required=False, default=None, type=argparse.FileType('r'), 
		help='Input [gzip] blank strip results files (default: %(default)s)'
	)
	parser.add_argument('--debug', 
		required=False, action='store_true', 
		help='Print DEBUG info (default: %(default)s)'
	)
	args = parser.parse_args()
	
	## Set up basic debugger
	logFormat = "[%(levelname)s]: %(message)s"
	logging.basicConfig(format=logFormat, stream=sys.stderr, level=logging.INFO)
	if args.debug:
		logging.getLogger().setLevel(logging.DEBUG)
	
	logging.debug('%s', args) ## DEBUG
	
	combine_results(args.test_results, args.blank_results, args.out)



def combine_results(test_results, blank_results, out):
	# Header
	t = 'Files\ttype'
	for name in NAMES:
		t = t + '\t' + name + '_intensity\t' + name + '_REA'
	out.write(t + '\n')

	# Blank strips
	blank = load_results(blank_results)
	t = blank_results.name + '\tblank'
	for name in NAMES:
		t = t + '\t' + str(blank[name]) + '\tNA'
	out.write(t + '\n')

	# Test strips
	for test_file in test_results:
		results, REA = load_and_process_test_results(test_file, blank)
		t = test_file.name + '\ttest'
		for name in NAMES:
			t = t + '\t' + str(results[name]) + '\t' + str(REA[name])
		out.write(t + '\n')



NAMES = ["Leukocytes", "Glucose", "Ketone", "Blood"]



def load_and_process_test_results(results_file, blank):
	results = load_results(results_file)
	REA = {}
	for name in NAMES:
		try:
			REA[name] = blank[name] - results[name]
		except TypeError:
			REA[name] = results[name]
	return results, REA



def load_results(results_file):
	results = {x:255.0 for x in NAMES}
	if results_file is not None:
		with results_file as r:
			for line in r:
				line = line.strip().split('\t')
				if line[0] in NAMES:
					try:
						results[line[0]] = float(line[1])
					except ValueError:
						results[line[0]] = line[1]
		
		# Assume if results equals 255 assume that this name was missing from the results file.
		for name in NAMES:
			if results[name] == 255.0:
				logging.warning('A value for %s was not found in %s results file. Setting to 255 as default.', name, results_file.name)
	logging.debug('%s', results) ## DEBUG
	return results



if __name__ == '__main__':
	main()
