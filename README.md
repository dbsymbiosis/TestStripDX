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

Copy and paste your custom .weights file into the 'data' folder and copy and paste your custom .names into the 'models/' folder.



Convert yolov4 detector to Tensorflow detector

```bash
./TestStripDX.py convert -m URS10
```

Crop and save target areas as new images

```bash
./TestStripDX.py process -m URS10 -v video1.mp4 video2.mp4 video3.mp4 .. ..
```

A number of output files (and a directory with temp files) for each input video file will be created by this command and will have the suffix `*.TestStripDX`

Combine the `*.results.txt` files produced by the previous command together into a single file.
Will also produce REA values. If a blank sample is prodided this will be used for REA calcualtion, otherwise a value of 255 will be used.
```bash
./TestStripDX.py combine -m URS10 -b video1.mp4.TestStripDX.results.txt -t video2.mp4.TestStripDX.results.txt video3.mp4.TestStripDX.results.txt -o combined_results.txt
```

Test TestStripDX.
Predict, predict and crop images.![3c7df9efa8480c71d55df8defe897db](https://user-images.githubusercontent.com/99760789/156899115-35268c08-938d-4c40-8d95-a781382dfe52.png)

Note: The showing labels are not related to the actual reagents, but the showing labels are exact same for each images. So, we correct this in .m file. We will retrain the model to try to correct this error.

Measure RGB values.
  ![4eb2e7d3cee213a42dfdbd4567ca0c9](https://user-images.githubusercontent.com/99760789/156899174-25a657f6-9c7c-4c9b-b394-28e9b76d6a49.png)





