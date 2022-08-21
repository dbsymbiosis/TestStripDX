#!/usr/bin/env python
DESCRIPTION = '''
Takes a list of PDF files (either from command line or from stdin) and merges them into a single multipage document

NOTE:
	- Depending on the PDFs being merged this script might produce a few warnings.
	  Nothing we can do to fix these problems (it has to do with how the PDFs are formed) so just ignore them. 
	
	PdfReadWarning: Multiple definitions in dictionary at byte 0x1f0e for key /F3 [generic.py:588]
	PdfReadWarning: Multiple definitions in dictionary at byte 0x1f18 for key /F4 [generic.py:588]
	...

'''
import sys
import os
import argparse
import logging
from PyPDF2 import PdfFileMerger

## Pass arguments.
def main():
	## Pass command line arguments. 
	parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=DESCRIPTION)
	parser.add_argument('-i', '--infiles', nargs='+', metavar="file.pdf", 
		required=False, default=sys.stdin, type=str, 
		help='Input pdf files (default: stdin)'
	)
	parser.add_argument('-o', '--out', metavar='merged.pdf', 
		required=True, type=str, 
		help='Output merged pdf file.'
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
	
	merge_PDFs(args.infiles, args.out)



def merge_PDFs(input_PDFs, output_file):
	## See https://stackoverflow.com/questions/3444645/merge-pdf-files
	merger = PdfFileMerger(strict=False)
	
	for pdf in input_PDFs:
		merger.append(pdf)
	
	merger.write(output_file)
	merger.close()



if __name__ == '__main__':
	main()
