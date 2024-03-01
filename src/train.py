import logging
import sys
import torch

from roboflow import Roboflow
from ultralytics import YOLO

from src.Utilities.constants import YOLOV8_models


def device() -> str:
    if torch.cuda.is_available():
        return 'cuda:0'
    return 'cpu'


def train(api_key: str, workspace_name: str, project_name: str, model_type: str, version: int, num_of_epochs: int = 10,
          tune: bool = False):
    if torch.cuda.is_available():
        logging.info(f'GPU is available. Loading the model to GPU...')
        dev = 'cuda:0'
    else:
        logging.info(f'No GPU available. So we are using the CPU for training purposes')
        dev = 'cpu'
    device = torch.device(dev)
    rf = Roboflow(api_key)
    project = rf.workspace(workspace_name).project(project_name)
    dataset = project.version(version).download('yolov8')
    if model_type not in YOLOV8_models:
        logging.error(f'Invalid model type: {model_type}. Model type should be one of the yolov8 models i. (yolov8n,'
                      f'yolov8s,yolov8m,yolov8l,yolov8x)')
        sys.exit(1)
    model = YOLO(model_type)
    model = model.to(device)
    datasetloc = dataset.location + '/data.yaml'
    logging.info(f'Dataset location:{datasetloc}')
    if tune:
        tuning_results = model.tune(task='detect',data=datasetloc, epochs=num_of_epochs, imgsz=640, plots=True, iterations=200)
    else:
        training_results = model.train(task='detect', data=datasetloc, epochs=num_of_epochs, imgsz=640, plots=True,
                                       optimizer='AdamW', lr0=0.00871,lrf=0.01629,momentum= 0.87533,
                                       weight_decay= 0.00034,warmup_epochs= 3.52483,warmup_momentum= 0.69684,
                                       box= 8.32671, cls= 0.40148,dfl= 2.23285,hsv_h= 0.01509, hsv_s= 0.6336
                                       ,hsv_v= 0.35286,degrees= 0.0,translate= 0.08973,scale= 0.59123,shear= 0.0
                                       ,perspective= 0.0,flipud= 0.0,fliplr= 0.39051,mosaic= 0.90573,mixup= 0.0,
                                       copy_paste= 0.0,save=True, val=True)
