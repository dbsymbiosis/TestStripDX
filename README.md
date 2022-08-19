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
  python save_model.py
  ```

  Crop and save target areas as new images

```bash
# Open an Image
img = Image.open('test/VID_20220817_152249.mp4.TestStripDX/frame.30sec.detect.detection1.png')
 
# Call draw Method to add 2D graphics in an image
I1 = ImageDraw.Draw(img)
 
# Add Text to an image
I1.text((28, 36), "nice Car", fill=(255, 0, 0))
 
# Display edited image
img.show()
 
# Save the edited image
img.save("t.png")
```

Test TestStripDX.
Predict, predict and crop images.![3c7df9efa8480c71d55df8defe897db](https://user-images.githubusercontent.com/99760789/156899115-35268c08-938d-4c40-8d95-a781382dfe52.png)

Note: The showing labels are not related to the actual reagents, but the showing labels are exact same for each images. So, we correct this in .m file. We will retrain the model to try to correct this error.

Measure RGB values.
  ![4eb2e7d3cee213a42dfdbd4567ca0c9](https://user-images.githubusercontent.com/99760789/156899174-25a657f6-9c7c-4c9b-b394-28e9b76d6a49.png)





