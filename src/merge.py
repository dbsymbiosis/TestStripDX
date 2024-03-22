import sys
import os
import logging
from PyPDF2 import PdfFileMerger, PdfMerger
from os import path
from glob import glob


# Combine results from TestStripDX for each strip/video.
#
#	test_files	results files from test videos
#	outfile		output combined results file
#	intervals	time intervals to collect frames from videos
def combine_results(test_files, outfile, intervals):
    logging.info('####')  ## INFO
    logging.info('#### Combining results files')  ## INFO
    logging.info('####')  ## INFO

    logging.info('Combining results files')  ## INFO

    ## Extract names and times to extract from video.
    names = []
    for x in intervals:
        names.append(x[0] + '_score')
        names.append(x[0] + '_R')
        names.append(x[0] + '_G')
        names.append(x[0] + '_B')

    ## Header
    t = 'Files'
    for name in names:
        t = t + '\t' + name
    outfile.write(t + '\n')

    ## Test strips
    for test_file in test_files:
        logging.info('Loading results file: %s', test_file.name)  ## INFO
        results = load_results(test_file, names)
        t = test_file.name
        for name in names:
            t = t + '\t' + str(results[name])
        outfile.write(t + '\n')

    logging.info('####')  ## INFO
    logging.info('#### Finished combining results files')  ## INFO
    logging.info('####')  ## INFO


def load_results(results_file, names, delimiter='\t'):
    results = {x: 'NA' for x in names}

    ## Check that results file exists
    if not os.path.exists(results_file.name):
        logging.error('Results file (%s) does not exist!', results_file.name)  ## ERROR
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
    logging.debug('%s', results)  ## DEBUG

    # Check that all names have values.
    for name in names:
        if results[name] == 'NA':
            logging.warning('A value for %s was not found in %s results file.', name, results_file.name)

    return results


# Takes a list of PDF files (either from command line or from stdin) and merges them into a single multipage document
#
# 	input_PDFs	input PDF files
#	output_file	outout combined PDF file
#	dir_path join all pdf files within folder
def joinPDFs(output_file, input_PDFs=None, dir_path: str = None):
    logging.info('####')  ## INFO
    logging.info('#### Merging PDF files')  ## INFO
    logging.info('####')  ## INFO
    if dir_path:
        input_PDFs = glob(path.join(dir_path, "*.{}".format('pdf')))
    merger = PdfMerger(strict=False)

    for pdf in input_PDFs:
        logging.info('Merging PDF file: %s', pdf)  ## INFO
        merger.append(pdf)

    merger.write(output_file)
    merger.close()
    logging.info('####')  ## INFO
    logging.info('#### Finished merging PDF files')  ## INFO
    logging.info('####')  ## INFO
