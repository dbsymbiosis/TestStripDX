% coral strips rgb measurement
%Glucose
Glucose = imread ('C:\Repos\yolov4-custom-functions\detections\crop\High_PC_093_TRIS1\Urobilinogen_1.png');

r1 = Glucose(:,:,1);
g1 = Glucose(:,:,2);
b1 = Glucose(:,:,3)
rglucose = mean(r1(:));
gglucose = mean(g1(:));
bglucose = mean(b1(:));
avgglucose = (rglucose+gglucose+bglucose)/3;

%Ketone
Ketone = imread ('C:\Repos\yolov4-custom-functions\detections\crop\High_PC_093_TRIS2\Protein_1.png');
r2 = Ketone(:,:,1);
g2 = Ketone(:,:,2);
b2 = Ketone(:,:,3);
rketone = mean(r2(:));
gketone = mean(g2(:));
bketone = mean(b2(:));
avgketone = (rketone+gketone+bketone)/3;

%Blood
Blood = imread ('C:\Repos\yolov4-custom-functions\detections\crop\High_PC_093_TRIS3\Nitrite_1.png');
r3 = Blood(:,:,1);
g3 = Blood(:,:,2);
b3 = Blood(:,:,3);
rblood = mean(r3(:));
gblood = mean(g3(:));
bblood = mean(b3(:));
avgblood = (rblood+gblood+bblood)/3;

%Leukocytes
Leukocytes = imread ('C:\Repos\yolov4-custom-functions\detections\crop\High_PC_093_TRIS4\pH_1.png');
r4 = Leukocytes(:,:,1);
g4 = Leukocytes(:,:,2);
b4 = Leukocytes(:,:,3);
rleukocytes = mean(r4(:));
gleukocytes = mean(g4(:));
bleukocytes = mean(b4(:));
avgleukocytes = (rleukocytes+gleukocytes+bleukocytes)/3;

fprintf('avgglucose=%f',avgglucose);
fprintf('avgketone=%f', avgketone);
fprintf('avgblood=%f', avgblood);
fprintf('avgleukocytes=%f', avgleukocytes);


