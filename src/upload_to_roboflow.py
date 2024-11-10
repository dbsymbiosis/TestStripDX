import os
import logging
import shutil
import sys

from roboflow import Roboflow
from ultralytics import YOLO
#from ultralytics.engine.results import Boxes

#from src.Utilities.color_space_values import color_space_values
#from src.detector import detect_test_strip
#from src.utils import shift_hue



def process_videos_and_upload_to_roboflow(videos, model_detector_path, test_analysis_times, outdir_suffix,
                                          api_key, project_name, output_txt_file_path):
    logging.info('####')  ## INFO
    logging.info('#### Processing video files')  ## INFO
    logging.info('####')  ## INFO

    ## Times to extract from video - make unique and sort.
    times = sorted(set([x[1] for x in test_analysis_times]))

    ## Process each video.
    for video in videos:
        logging.info('# Extracting frames from %s', video)  ## INFO

        ## Envs
        outdir = video + outdir_suffix
        os.mkdir(outdir)
        frame_prefix = os.path.join(outdir, "frame")

        ## Check if video file exists.
        if not os.path.exists(video):
            logging.error('Video file %s does not exists!', video)  ## ERROR
            sys.exit(1)

        ## Extract frame from a specific timestamp in a video.
        capture_frames_from_video(video, frame_prefix, times)

        ## Crop tests from each time frame
        for time in times:
            frame_in = frame_prefix + "." + str(time) + "sec.png"
            logging.info('Searching for tests in time %s seconds image', time)  ## INFO
            logging.debug('In frame: %s', frame_in)  ## DEBUG

            predict_image_save_boxes(frame_in, model_detector_path, output_txt_file_path, api_key, project_name)



def predict_image_save_boxes(image_path: str, model_path: str, output_text_path: str, roboflow_api_key: str,
                             project_name: str):
    logging.info(model_path)
    model = YOLO(model_path)
    logging.info('Hi')
    predictions = model.predict(source=image_path, stream=False)
    logging.info(f'output dir: {output_text_path}')
    open(output_text_path, 'w').close()
    with open(output_text_path, '+w') as file:
        logging.error(f'Opened file: {output_text_path}')
        for idx, prediction in enumerate(predictions[0].boxes.xywhn):  # change final attribute to desired box format
            cls = int(predictions[0].boxes.cls[idx].item())
            # path = predictions[0].path
            class_name = model.names[cls]
            logging.error(class_name)
            file.write(
                f"{cls} {prediction[0].item()} {prediction[1].item()} {prediction[2].item()} {prediction[3].item()}\n")
    rf = Roboflow(api_key=roboflow_api_key)
    project = rf.workspace().project(project_name)
    logging.info(
        project.upload(image_path=image_path, annotation_path=output_text_path, split='train'))



