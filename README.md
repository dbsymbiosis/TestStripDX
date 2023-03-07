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


Download files (not yet uploaded).
```bash
wget 
wget 
wegt
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

For each of your input videos there should also be a PDF file created called `*.TestStripDX.detection.pdf`. This document is simply a combination/concatenation of the images extracted from the video that was processed, and that were used to derive the intensity values. Its main use is to allow easy double checking of the computer vision component of the workflow (i.e., to double check that the correct tests were identified by the vision system at the correct time points).

Ideally, your images should look similar (with maybe some boxes missing or orders swapped) to this image.
![3c7df9efa8480c71d55df8defe897db](https://user-images.githubusercontent.com/99760789/156899115-35268c08-938d-4c40-8d95-a781382dfe52.png)


If you wish to combine the `*.TestStripDX.detection.pdf` files together to make it easier to scan through, use the following command.
```bash
./TestStripDX.py joinPDFs -o merged.pdf -i *.TestStripDX.detection.pdf
```


