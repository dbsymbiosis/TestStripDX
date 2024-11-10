#!/usr/bin/env python3
import sys
import os
import argparse
from argparse import RawTextHelpFormatter
import logging
import subprocess
import torch

from src.common import get_test_analysis_times
from src.graph import gen_save_group_chart_from_csv
from src.video import process_videos


def execute_commands():
    # TODO: too long function, divide into meaningful functions
    import os
    import logging
    import sys
    from src.merge import joinPDFs
    from src.common import get_test_analysis_times
    from src.video import process_videos
    from src.merge import combine_results
    from src.extract import extract
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
                                      required=False, nargs='+', type=str,
                                      help='Video files to process'
                                      )
    parser_process_video.add_argument('-ip', '--in_videos_dir',
                                      required=False, type=str,default='',
                                      help='Path to the directory containing the videos to be processed'
                                      )
    parser_process_video.add_argument('-m', '--model', metavar='model_name',
                                      required=False, type=str, default='URS10',
                                      help='Name of test strip being run. (default: %(default)s). '
                                           'Must have downloaded model files in models/ directory.'
                                      )
    parser_process_video.add_argument('-t', '--tests', metavar='tests', nargs='*',
                                      required=False, type=str, default=[],
                                      help='List of the tests to be run within the video. '
                                           'Provide the test names with spaces. Eg: -tests Glucose Blood'
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
    parser_joinPDFs.add_argument('-d', '--dir', metavar='',
                                 required=False, type=str,
                                 help='Path to directory containing the pdf files to be merged')
    
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
    ## Parser for command to join PDFs
    ##
    GENERATE_GRAPHS_FROM_CSV = '''
    
    Extract data from a csv file based on the column names given, and generate a grouped bar graph using the
    extracted data.
    
    '''
    parser_gen_graph = subparsers.add_parser('gen_graph',
                                             help='Generate graph from a csv file',
                                             description=GENERATE_GRAPHS_FROM_CSV,
                                             formatter_class=RawTextHelpFormatter
                                             )
    parser_gen_graph.add_argument('-p', '--path',
                                  required=True, type=str, default='Results.csv',
                                  help='Path to the csv file'
                                  )
    parser_gen_graph.add_argument('-x', '--x_label',
                                  required=True, type=str, default='Video-Name',
                                  help='Name of the column used for x axis'
                                  )
    parser_gen_graph.add_argument('-y', '--y_labels',
                                  required=True, type=str, default='TEST-BILIRUBIN-Red.shift0', nargs='+',
                                  help='List of names of columns used for generating bars within the graph'
                                  )
    parser_gen_graph.add_argument('-o', '--output-dir', required=True, type=str,
                                  help='Path of the directory where you want to save the graph')
    parser_gen_graph.add_argument('-gh', '--graph_height', required=False, type=int,
                                  default=16,
                                  help='Desired height of the saved graph')
    parser_gen_graph.add_argument('-gw', '--graph_width', required=False, type=int,
                                  default=24,
                                  help='Desired width of the saved graph')
    parser_gen_graph.add_argument('--debug',
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
                                                              description=PROCESS_VIDEOS_UPLOAD_DESCRIPTION
                                                             )
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
                                                   help='Api-key to access the roboflow project'
                                                  )
    parser_process_video_and_roboflow.add_argument('-p', '--project',
                                                   required=True, type=str, default='',
                                                   help='Roboflow project name, where the annotations have to be uploaded to'
                                                  )
    parser_process_video_and_roboflow.add_argument('-o', '--output_text_path', metavar='output_text_path',
                                                   required=True, type=str,
                                                   help='Path to the output text file, which the program can use to save bounding boxes data'
                                                  )
    parser_process_video_and_roboflow.add_argument('--debug',
                                                   required=False, action='store_true',
                                                   help='Print DEBUG info (default: %(default)s)'
                                                   )
    
    ##
    ## Parser for training a model on a roboflow dataset
    ##
    TRAIN_ROBOFLOW_DATASET_DESCRIPTION = '''
    
    Runs training epochs on a YOLO model mentioned in the command line args and a dataset from 
    roboflow based on the api_key and project name mentioned in the command line arguments.
    
    '''
    parser_train_yolo_model = subparsers.add_parser('train',
                                                    help='Train YOLO model on a roboflow dataset',
                                                    description=TRAIN_ROBOFLOW_DATASET_DESCRIPTION)
    parser_train_yolo_model.add_argument('-m', '--model',
                                         required=True, type=str,
                                         help='Type of base YOLO model to be trained. Example: yolov8n, yolov8s'
                                        )
    parser_train_yolo_model.add_argument('-ak', '--apikey', required=True, type=str,
                                         help='Api_key to access the roboflow dataset'
                                        )
    parser_train_yolo_model.add_argument('-w', '--workspace', required=True, type=str,
                                         help='Workspace which contains the dataset on roboflow'
                                        )
    parser_train_yolo_model.add_argument('-p', '--project', required=True, type=str,
                                         help='Name of the project on Roboflow'
                                        )
    parser_train_yolo_model.add_argument('-e', '--epochs', required=False, type=int,
                                         metavar='10', help='Number of training epochs to run'
                                        )
    parser_train_yolo_model.add_argument('-t', '--tune', required=False, action='store_true',
                                         help='Enable hyperparameter tuning of the model on default set of variables and values'
                                        )
    parser_train_yolo_model.add_argument('-pt', '--partial_trained',required=False, type=str,
                                         metavar='',help='Path to the partially trained model to resume training from.'
                                        )
    parser_train_yolo_model.add_argument('-v', '--version', required=True, type=int,
                                         help='Version of the roboflow project to train on'
                                        )
    parser_train_yolo_model.add_argument('--debug',
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
    model_detector_path = ''
    if args.command in ['train', 'process', 'combine', 'predict_and_upload_to_roboflow']:
        model_detector_path = os.path.join(script_dir, models_dir, args.model + '.pt')
        
        ## Check model files exist
        logging.info('Checking model files (%s/%s.*) exist', models_dir, args.model)  ## INFO
        for file_path in [model_detector_path]:
            if not os.path.exists(file_path):
                logging.error('Model file (%s) does not exist!', file_path)  ## ERROR
                sys.exit(1)
    
    ## Model variables
    ## Import model params
    TEST_ANALYSIS_TIMES = []
    try:
        TEST_ANALYSIS_TIMES = get_test_analysis_times(args.tests)
    except:
        logging.error('No test arguments passed')
        # if args.command == #TODO: throw error when tests is required but not given(maybe make tests mandatory in the add_argument command)
    logging.debug(f'Test Analysis times: {TEST_ANALYSIS_TIMES}')  # DEBUG
    ## Extract just the times from list of test names and times.
    times = sorted(set([x for x in TEST_ANALYSIS_TIMES]))

    if args.command in ['extract']:
        times = []
        if args.times != None:
            times = args.times
        logging.debug('seconds: %s', times)  ## DEBUG

    ## Run subcommand
    #	NOTE: Import each set of functions as needed becuase many of the packages take >30 sec to import
    #	      so we need to only run import when we need to
    if args.command == 'process':
        process_videos(args.in_videos,
                       model_detector_path,
                       times,  # Timings based on the tests input from the command line argument
                       args.cleanup, args.suffix,in_vid_dir=args.in_videos_dir)
    elif args.command == 'combine':
        combine_results(args.in_results, args.out_combined, TEST_ANALYSIS_TIMES)
    elif args.command == 'joinPDFs':
        if args.dir and args.dir != '':
            joinPDFs(output_file=args.out_pdf, dir_path=args.dir)
        else:
            joinPDFs(output_file=args.out_pdf, input_PDFs=args.in_pdfs)
    elif args.command == 'extract':
        extract(args.in_videos, args.outdir, sorted(set([x[1] for x in TEST_ANALYSIS_TIMES])))
    elif args.command == 'train':
        train(args.apikey, args.workspace, args.project, args.model, args.version, args.epochs, args.tune, args.partial_trained)
    elif args.command == 'gen_graph':
        gen_save_group_chart_from_csv(args.path, args.x_label, args.y_labels, args.output_dir, args.graph_height,
                                      args.graph_width)
    
    # Roboflow upload and training
    elif args.command == 'predict_and_upload_to_roboflow':
        from src.upload_to_roboflow import process_videos_and_upload_to_roboflow
        process_videos_and_upload_to_roboflow(args.in_videos, model_detector_path, times,
                                              args.suffix, args.apikey, args.project, args.output_text_path)
    elif args.command == 'train':
        from src.train import train
        train(args.apikey, args.workspace, args.project, args.model, args.version, args.epochs, args.tune, args.partial_trained)



    logFormat = "[%(levelname)s]: %(message)s"
    logging.basicConfig(format=logFormat, stream=sys.stderr, level=logging.INFO)


if __name__ == '__main__':
    execute_commands()
