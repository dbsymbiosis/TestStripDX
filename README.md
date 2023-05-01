# TestStripDX

TestStripDX: An image processing framework for processing and extracting test strip results from a photo.
This repo walks you through how to analyze videos of test strips.

## Install `TestStripDX`

Download the `TestStripDX` GitHub repository.

```bash
git clone https://github.com/dbsymbiosis/TestStripDX.git
```

Set up Conda environment.
We recommend to download Anaconda to set up tensorflow environment.
```bash
conda env create -f conda.yml
conda activate TestStripDX
```

Crop and save target areas as new images
```bash
./TestStripDX.py process -i video1.mp4 video2.mp4 video3.mp4 .. ..
```

A number of output files (and a directory with temp files) for each input video file will be created by this command and will have the suffix `*.TestStripDX`

Combine the `*.results.txt` files produced by the previous command together into a single file.
```bash
./TestStripDX.py combine -o combined_results.txt -i video2.mp4.TestStripDX.results.txt video3.mp4.TestStripDX.results.txt
```

For each of your input videos there should also be a PDF file created called `*.TestStripDX.detection.pdf`. This document is simply a combination/concatenation of the images extracted from the video that was processed, and that were used to derive the intensity values. Its main use is to allow easy double checking of the computer vision component of the workflow (i.e., to double check that the correct tests were identified by the vision system at the correct time points).

Ideally, your images should look similar (with maybe some boxes missing or orders swapped) to this image.
![3c7df9efa8480c71d55df8defe897db](https://user-images.githubusercontent.com/99760789/156899115-35268c08-938d-4c40-8d95-a781382dfe52.png)


If you wish to combine the `*.TestStripDX.detection.pdf` files together to make it easier to scan through, use the following command.
```bash
./TestStripDX.py joinPDFs -o merged.pdf -i *.TestStripDX.detection.pdf
```


