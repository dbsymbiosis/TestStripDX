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
Download the zipped dataset includes all images and related .txt files like shown in images folder: ![image](https://user-images.githubusercontent.com/99760789/156896891-97db7fcd-7b06-419c-838c-3493afcd882e.png)<br />
#### 1. Get started<br />
#### 2. Create new project (name it)<br />
#### 3. Annotate (group and name)<br />
To annotate, use the second square tool in right white bar to square the target area, then group and name every target areas. <br />
NOTE: Annotations are CASE SENSITIVE, so label all images used for training a model with the exact same labels. <br />
Annotation sample:![ed8f7710c01f4b22005a57eb09dc4ef](https://user-images.githubusercontent.com/99760789/156478377-41172c87-93b7-42b7-a5ca-9a9d479a781e.png)<br />
#### 4. Assign<br />
Assign images into train and valid datasets which are for training and validing the custom detector in 80%:20% ratio.<br />
![f0a4936550ebb131a5cf985d230dd0c](https://user-images.githubusercontent.com/99760789/156479422-732e1d7b-d7c1-45d2-9d44-8ffe5ba7e78e.png)<br />
#### 5. Generate dataset<br />
To generate dataset, in preprocessing section. We resized images in 416* 416 which can accelerate the training before downing annotated dataset: <br />
![32fff1758cb5304017ab60be2cb7dec](https://user-images.githubusercontent.com/99760789/156482596-06d385ad-003d-489d-b997-52949351b6c9.png)  <br /> <br />
For Augmentation, press Continue. <br /> 
While generating dataset, your screen will look like this: <br />
![generating loading screen](https://user-images.githubusercontent.com/77503347/187273187-eb2118a2-51c9-40f8-9a73-6462a6a67ee0.png)<br />
#### 6. Export and Download <br />
After generating the dataset, click on the 'Export' option to export and download your dataset. <br /> 
![export option](https://user-images.githubusercontent.com/77503347/187276282-9d194f83-3890-4f15-85a8-cb54a2ab78c1.png) <br /> 
In the pop-up dialog box, select 'YOLO Darknet' format and 'Download zip to computer' option. <br />
![export settings](https://user-images.githubusercontent.com/77503347/187276697-09bd6df6-8452-4d6a-a3c6-b277eb1931a2.png) <br /> 


Download the zipped dataset includes all images and related .txt files like shown in images folder: ![9d7e3ded18702bbafbc578ca574cb30](https://user-images.githubusercontent.com/99760789/156482401-83e1e3f8-ffbe-4194-bb4d-89a36f65fbbb.png)<br />

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
Download Tensorflow folder to local drive. We recommend to use Gitbash shell to deliver command and visual studio code as editor.<br />
1. Set up Conda environment<br />
We recommend to download Anaconda to set up tensorflow environment. Then deliver command in Gitbash shell to create and activate GPU or CPU.<br />

Tensorflow CPU<br />
conda env create -f conda-cpu.yml<br />
conda activate yolov4-cpu<br />

Tensorflow GPU<br />
conda env create -f conda-gpu.yml<br />
conda activate yolov4-gpu<br />

2. Download 'yolov4-obj_best.weights' file from backup folder.<br />

3. Use custom trained detector<br />

Copy and paste your custom .weights file into the 'data' folder and copy and paste your custom .names into the 'data/classes/' folder.<br />

The only change within the code you need to make in order for your custom model to work is on line 14 of 'core/config.py' file. Update the code to point at your custom .names file as seen below. (my custom .names file is called custom.names but yours might be named differently)<br />
![image](https://user-images.githubusercontent.com/99760789/156898001-df800ec3-0478-44ad-8ffc-82f9b6f14920.png)<br />

4. Convert yolov4 detector to Tensorflow detector<br />
python save_model.py --weights ./data/yolov4.weights --output ./checkpoints/yolov4-416 --input_size 416 --model yolov4 <br />
Paste this command into Gitbash.<br />

5. Crop and save target areas as new images<br />
python detect.py --weights ./checkpoints/yolov4-416 --size 416 --model yolov4 --images ./data/images/'your image name'.jpg --crop<br />
Imput this command into Gitbash, make sure replace'your image name' to your image name.<br />
6. Measure RGB values in Matlab.<br />
Use .m MATLAB file to measure cropped images, make sure to use correct directory.<br />
![image](https://user-images.githubusercontent.com/99760789/156898578-0350354d-71fa-4aa9-8eb9-e1885a128318.png)<br />
7. Test TestStripDX.<br />
Predict, predict and crop images.<br />
![3c7df9efa8480c71d55df8defe897db](https://user-images.githubusercontent.com/99760789/156899115-35268c08-938d-4c40-8d95-a781382dfe52.png)<br />
Note: The showing labels are not related to the actual reagents, but the showing labels are exact same for each images. So, we correct this in jupyter file. We will retrain the model to try to correct this error.<br />
Measure RGB values.<br />
The last step is to run the TeststripsDX file in Jupyter Note. Then, it will create an excel file. Please check the TeststripsDX folder for details including the code, result file, and testing dataset.<br />
![8858ed4a3c1c72014144248a3c119ac](https://user-images.githubusercontent.com/99760789/182762850-bf7a90cf-8ff4-4efd-972c-f9bd35d28a17.jpg)


















