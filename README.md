# TestStripDX

TestStripDX: An image processing framework for processing and extracting test strip results from a photo.
This repo walks you through how to analyze videos of test strips using YOLOv4 and Tensorflow.

![3c7df9efa8480c71d55df8defe897db](https://user-images.githubusercontent.com/99760789/156899115-35268c08-938d-4c40-8d95-a781382dfe52.png)

## Install `TestStripDX`

Download the `TestStripDX` GitHub repository.

```bash
git clone https://github.com/dbsymbiosis/TestStripDX.git
```

Set up Conda environment.
We recommend to download Anaconda to set up tensorflow environment.

Setup Tensorflow for CPU
```bash
conda env create -f conda-cpu.yml
conda activate yolov4-cpu
```

Setup Tensorflow for GPU
```bash
conda env create -f conda-gpu.yml
conda activate yolov4-gpu
```

## Deploy .weights file into the Tensorflow

We now need to download and setup the weights files that are used by `TestStripDX` for its analysis. 
Download files.
```bash
wget 
wget 
```

Copy and paste your custom .weights file into the 'data' folder and copy and paste your custom .names into the 'data/classes/' folder.<br />



4. Convert yolov4 detector to Tensorflow detector<br />

  ```bash
  python save_model.py --weights ./data/yolov4.weights --output ./checkpoints/yolov4-416 --input_size 416 --model yolov4
  ```

  

5. Crop and save target areas as new images<br />
  python detect.py --weights ./checkpoints/yolov4-416 --size 416 --model yolov4 --images ./data/images/'your image name'.jpg --crop<br />
  Imput this command into Gitbash, make sure replace'your image name' to your image name.<br />

6. Test TestStripDX.<br />
  Predict, predict and crop images.<br />
  ![3c7df9efa8480c71d55df8defe897db](https://user-images.githubusercontent.com/99760789/156899115-35268c08-938d-4c40-8d95-a781382dfe52.png)<br />
  Note: The showing labels are not related to the actual reagents, but the showing labels are exact same for each images. So, we correct this in .m file. We will retrain the model to try to correct this error.<br />
  Measure RGB values.<br />
  ![4eb2e7d3cee213a42dfdbd4567ca0c9](https://user-images.githubusercontent.com/99760789/156899174-25a657f6-9c7c-4c9b-b394-28e9b76d6a49.png)





