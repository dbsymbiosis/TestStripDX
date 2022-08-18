#!/usr/bin/env python3
DESCRIPTION = '''
TestStripDX

An image processing framework for processing and extracting test strip results from a photo.
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
	parser.add_argument('--debug',
		required=False, action='store_true',
		help='Print DEBUG info (default: %(default)s)'
	)
	subparsers = parser.add_subparsers(dest='command')
	
	# Parser for the conversion of the yolov4 to Tensorflow detector
	parser_convert_detector = subparsers.add_parser('convert', help='Convert yolov4 detector to Tensorflow detector')
	parser_convert_detector.add_argument('-w', '--weights', metavar='model.weights', 
		required=False, default="models/teststrips.weights", type=str, 
		help='Path to weights file (default: %(default)s)'
	)
	parser_convert_detector.add_argument('-o', '--out', metavar='model.yolov4-416', 
		required=False, default="models/teststrips.yolov4-416", type=str, 
		help='Path to output (default: %(default)s)'
	)
	
	# Parser for the processing of the test strip video files
	parser_process_video = subparsers.add_parser('process', help='Process test strip video files')
	parser_process_video.add_argument('-v', '--videos', metavar='teststrip.mp4', 
		required=True, nargs='+', type=str, 
		help='Video files to process'
	)
	
	# Parser for the combining of the results files into a single output
	parser_combine_results = subparsers.add_parser('combine', help='Combine results from processed video files')
	parser_combine_results.add_argument('-t', '--test_results', metavar='test_results.txt',
		required=True, nargs='+', type=argparse.FileType('r'),
		help='Input [gzip] test strip results files (required)'
	)
	parser_combine_results.add_argument('-o', '--out', metavar='output.txt',
		required=False, default=sys.stdout, type=argparse.FileType('w'),
		help='Output [gzip] file (default: stdout)'
	)
	parser_combine_results.add_argument('-b', '--blank_results', metavar='blank_results.txt',
		required=False, default=None, type=argparse.FileType('r'),
		help='Input [gzip] blank strip results files (default: not used)'
	)
	
	# Parse all arguments.
	args = parser.parse_args()
	
	## Set up basic debugger
	logFormat = "[%(levelname)s]: %(message)s"
	logging.basicConfig(format=logFormat, stream=sys.stderr, level=logging.INFO)
	if args.debug:
		logging.getLogger().setLevel(logging.DEBUG)
	
	logging.info('%s', args) ## DEBUG
	
	
	#with args.input as infile, args.out as outfile:
	#	for line in infile:
	#		print line.strip()
	#	print "Done printing"
	
	
	## Best to just use a with statement but if you need to operate on the object directly
	
	## Loop over file (need to uncomment __iter__() and close() in File class)
	#for line in args.input:
	#	print line.strip()
	#args.input.close()
	
	## Close file handles (need to uncomment close() method in File class)
	#args.input.close()
	#args.out.close()
	#if args.bam is not None:
	#	args.bam.close()
	#if args.info is not None:
	#	args.info.close()



class File(object):
	'''
	Context Manager class for opening stdin/stdout/normal/gzip files.

	 - Will check that file exists if mode='r'
	 - Will open using either normal open() or gzip.open() if *.gz extension detected.
	 - Designed to be handled by a 'with' statement (other wise __enter__() method wont 
	    be run and the file handle wont be returned)
	
	NOTE:
		- Can't use .close() directly on this class unless you uncomment the close() method
		- Can't use this class with a 'for' loop unless you uncomment the __iter__() method
			- In this case you should also uncomment the close() method as a 'for'
			   loop does not automatically cloase files, so you will have to do this 
			   manually.
		- __iter__() and close() are commented out by default as it is better to use a 'with' 
		   statement instead as it will automatically close files when finished/an exception 
		   occures. 
		- Without __iter__() and close() this object will return an error when directly closed 
		   or you attempt to use it with a 'for' loop. This is to force the use of a 'with' 
		   statement instead. 
	
	Code based off of context manager tutorial from: https://book.pythontips.com/en/latest/context_managers.html
	'''
	def __init__(self, file_name, mode):
		## Upon initializing class open file (using gzip if needed)
		self.file_name = file_name
		self.mode = mode
		
		## Check file exists if mode='r'
		if not os.path.exists(self.file_name) and mode == 'r':
			raise argparse.ArgumentTypeError("The file %s does not exist!" % self.file_name)
	
		## Open with gzip if it has the *.gz extension, else open normally (including stdin)
		try:
			if self.file_name.endswith(".gz"):
				#print "Opening gzip compressed file (mode: %s): %s" % (self.mode, self.file_name) ## DEBUG
				self.file_obj = gzip.open(self.file_name, self.mode+'b')
			else:
				#print "Opening normal file (mode: %s): %s" % (self.mode, self.file_name) ## DEBUG
				self.file_obj = open(self.file_name, self.mode)
		except IOError as e:
			raise argparse.ArgumentTypeError('%s' % e)
	def __enter__(self):
		## Run When 'with' statement uses this class.
		#print "__enter__: %s" % (self.file_name) ## DEBUG
		return self.file_obj
	def __exit__(self, type, value, traceback):
		## Run when 'with' statement is done with object. Either because file has been exhausted, we are done writing, or an error has been encountered.
		#print "__exit__: %s" % (self.file_name) ## DEBUG
		self.file_obj.close()
#	def __iter__(self):
#		## iter method need for class to work with 'for' loops
#		#print "__iter__: %s" % (self.file_name) ## DEBUG
#		return self.file_obj
#	def close(self):
#		## method to call .close() directly on object.
#		#print "close: %s" % (self.file_name) ## DEBUG
#		self.file_obj.close()


if __name__ == '__main__':
	main()
