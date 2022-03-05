# TestStripDX
TestStripDX: An image processing framework for processing and extracting test strip results from a photo.<br />
This repo walks you through how we prepare, train and run TestStripDX detector in the Cloud through Google Colab.<br />
![1_VluiXEpyenaqzuBkhtY3iA](https://user-images.githubusercontent.com/99760789/156474636-36180a09-8a37-4bbd-b76f-e8dd3e680ec1.jpeg)
YOLOv4 is a computer vision model for optimal speed and accuracy of object detection.<br />

## Prepare dataset for training
Before training custom detector, we need to prepare a dataset with annotations to provide your target areas to the model. Here, we utilized online annotation tool from the Roboflow which no needs to download and easy to use and save datasets: https://roboflow.com/annotate <br />
The dataset need to be as versatile as you can. For CoralDX, we utilized 40 pictures in images folder to train. And after annotating, the Roboflow will give a corresponding .txt file with the coordinates of your selected target areas.<br />

1. Get started<br />
2. Create new project (name it)<br />
3. Annotate (group and name)<br />
![eed6415d98702e9ff0b3778f4e7b269](https://user-images.githubusercontent.com/99760789/156896425-41ef0501-870f-4ddc-8442-670ae619b308.png)<br />
To annotate, use the second square tool in right white bar to square the target area, then group and name every target areas. <br />
4. Assign<br />
Assign images into train and valid datasets which are for training and validing the custom detector in 80%:20% ratio.<br />
![f0a4936550ebb131a5cf985d230dd0c](https://user-images.githubusercontent.com/99760789/156479422-732e1d7b-d7c1-45d2-9d44-8ffe5ba7e78e.png)<br />
5. Generate dataset<br />
To generate dataset, in preprocessing section. We resized images in 416* 416 which can accelerate the training before downing annotated dataset: <br />
![32fff1758cb5304017ab60be2cb7dec](https://user-images.githubusercontent.com/99760789/156482596-06d385ad-003d-489d-b997-52949351b6c9.png)  <br /> 
6. Download <br />
Download the zipped dataset includes all images and related .txt files like shown in images folder: ![55d4eac6799dd871bf918c90a54ce5f](https://user-images.githubusercontent.com/99760789/156896599-e20acd80-42f4-4b90-bafa-37c4f9504d11.png)<br />


## Train custom detector in Google Colab
[![Train Custom Model In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1_GdoqCJWXsChrOiY8sZMr_zbr_fH-0Fg?usp=sharing) Before start the journey, make a copy of this Colab file.
1. Enabling GPU within your notebook<br />
![d0d318308b8a04ba3a94d1ec327a3c7](https://user-images.githubusercontent.com/99760789/156487290-dd54f88f-1572-4df2-b004-7e634db5da36.png)
2. Cloning and Building Darknet<br />
![258e8e97c2ab3f8861b6bc6730b8291](https://user-images.githubusercontent.com/99760789/156487412-6400bfd3-1d38-436b-adba-bb933de8d56c.png)
3. Download pre-trained YOLOv4 weights<br />
![95b5950256875de9b8c8237f77b7194](https://user-images.githubusercontent.com/99760789/156487627-c606e03f-76d3-43fb-8731-fc182e6d09e1.png)
4. Define Helper Functions<br />
![201a368b6de66408c48dcc2d70825ad](https://user-images.githubusercontent.com/99760789/156487936-21a80d11-d28b-4c7b-9328-95ccc5e1e872.png)
5. Run Your Detections with Darknet and YOLOv4!<br />
![9366cf94f1b88e48147cb28657518ca](https://user-images.githubusercontent.com/99760789/156488011-5de69ac2-70f8-4a4d-81ab-bbe8636a0b9e.png)
6. Uploading Local or Google Drive Files to Use<br />
We recommend to create a Google Drive Folder called yolov4 and put enerything into Google Drive for use<br />
![6151825886ad39a3a628dc2449877fc](https://user-images.githubusercontent.com/99760789/156489104-b819ae2d-acbc-4e14-b72d-410f21aff1ea.png)<br />
The follwing list is the files need to upload into the Googlr Drive<br />
![df1eb079741e20d8bb341fbc6a7d2cf](https://user-images.githubusercontent.com/99760789/156489897-54d35a04-f711-444f-b662-4a42236a288a.png)<br />
Copy of YOLOv4.ipynb: copy of this Colab tutorial file<br />
images: images for test custom detector<br />
backup: create empty folder to store weights file<br />
obj.zip: change name of train folder to obj and compress<br />
test.zip: change name of valid folder to test and compress<br />
![13395d487ed5d8a7a537eb924252d64](https://user-images.githubusercontent.com/99760789/156493447-8e4e6f70-2fe6-4c84-a86f-036e7acfc8fa.png)<br />
yolov4-obj.cfg: configuration file<br />
![1ebdf19dc1b69652f6bc676f18fff58](https://user-images.githubusercontent.com/99760789/156493594-80522aa9-17d9-42bc-99f3-01adbc6494c5.png)<br />
obj.names: group names<br />
obj.data: directions of files<br />
Put group names in obj.names file and change the classes number for custom detector. Both file can be editedd from example files using Text Editor in cfg section.<br />
![3cd74b5b0b69b4fcd2297129bdbbed5](https://user-images.githubusercontent.com/99760789/156494828-17330600-7c08-44a0-a69e-e15771ad17d0.png)<br />
generate_train.py: configuration files to train our custom detector are the train.txt training images<br />
generate_test.py: configuration files to train our custom detector are the test.txt testing images<br />
![b58c4c568501705e3c31a3fa9d3d08d](https://user-images.githubusercontent.com/99760789/156495019-189c595f-f90f-41cc-80f0-4813498eb7fb.png)
classes.txt: group names<br />
Edit example file using Text Editor and put group names in.<br />
7. Start training<br />
![image](https://user-images.githubusercontent.com/99760789/156896712-82ebdbd1-cbc9-4d06-ad85-4af8ee86c634.png)<br />

## Deploy .weights file into the Tensorflow
