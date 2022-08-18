#!/usr/bin/env python
DESCRIPTION = '''
Extract frame from a specific timestamp in a video.
'''
import sys
import os
import argparse
import logging
import gzip
import sys
import numpy as np
import imageio

## Pass arguments.
def main():
	## Pass command line arguments. 
	parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=DESCRIPTION)
	parser.add_argument('-i', '--input_image', metavar='input.png', 
		required=True, type=str, 
		help='Input image file (required)'
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
	
	extract_colors(args.input_image, sys.stdout)



def extract_colors(image_filename, outfile):
	pic = imageio.imread(image_filename)
	R = pic[ :, :, 0]
	G = pic[ :, :, 1]
	B = pic[ :, :, 2]
	meanR = np.mean(R)
	meanG = np.mean(G)
	meanB = np.mean(B)
	outfile.write(str((meanR+meanG+meanB)/3)+'\n')



if __name__ == '__main__':
	main()
