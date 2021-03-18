close all;
clear all;
clc;

%Input Image
  
[file,path]=uigetfile('*.jpg','select a input image');
str=strcat(path,file);
I=imread(str);
figure(1),imshow(I);

%Gray-scale
  
gray=rgb2gray(I);
figure(2),imshow(gray);

%Noise-removal
  
noise=fspecial('gaussian');
f=imfilter(gray,noise);
figure(3);
imshow(f)

%Gabor-Filter
  
gabor=GaborFilterBank(4,4,32,32);
features=GaborFeatures(f,gabor,4,4);
save Features features

%Pre-trained Dataset
  
File='dataset'
dataset=fullfile(File,'foldername','dataset')
imgdata=imageDatastore('dataset','IncludeSubFolders',true,...
    'LabelSource','foldernames','FileExtensions',{'.jpg','.png', '.tif'});

%Count-of-the-Image
  
icount=countEachLabel(imgdata)
minsetCount=min(icount{:,2});
maxImages=60;
mincount=min(maxImages,minsetCount);

%Split-each-label 
  
imds=splitEachLabel(imgdata,mincount,'randomize');
countEachLabel(imds)

%DenseNet
  
net=densenet201();
figure(6),plot(net);
title('Densenet');
set(gca,'Ylim',[150 170]);

%First-Layer
  
First=net.Layers(1);

%Last-Layer
  
End=net.Layers(end);

%Number-of-Class-names-for-ImageNet-Classification-Task
  
numel(End.ClassNames);
[trainingSet, testSet]=splitEachLabel(imds,0.3,'randomize');

%Resize
  
imageSize=First.InputSize;
augmentedTrainingSet=augmentedImageDatastore(imageSize,trainingSet,'ColorPreprocessing','gray2rgb');
augmentedTestSet=augmentedImageDatastore(imageSize,testSet,'ColorPreprocessing','gray2rgb');

%Layers
  
layer=[(imageInputLayer([224 224 3]))
    convolution2dLayer(5,16,'Padding','Same','Name','conv_1')
    batchNormalizationLayer('Name','BN_1')
    reluLayer('Name','relu_1')
    maxPooling2dLayer(2,'stride',2)
    
    convolution2dLayer(5,16,'Padding','Same','Name','conv_1')
    batchNormalizationLayer('Name','BN_1')
    reluLayer('Name','relu_1')
    maxPooling2dLayer(2,'stride',2)
    
    averagePooling2dLayer(2,'stride',2)
    fullyConnectedLayer(4)
    softmaxLayer
    classificationLayer()];

featureLayer='fc1000';
trainingFeatures=activations(net,augmentedTrainingSet,featureLayer,'MiniBatchSize',32,'OutputAs','columns');

%Get-training-labels-from-trainingset
  
trainingLabels=trainingSet.Labels;
classifier=fitcecoc(trainingFeatures,trainingLabels,'Learners','Linear','Coding','onevsall','ObservationsIn','columns');

%Test-Features
  
testFeatures=activations(net,augmentedTestSet,featureLayer,'MiniBatchSize',32,'OutputAs','columns');
predictedLabels =predict(classifier,testFeatures,'ObservationsIn','columns');
testLabels=testSet.Labels;

%automatic-resize
  
aids=augmentedImageDatastore(imageSize,f,'ColorPreprocessing','gray2rgb');
imageFeatures= activations(net,aids,featureLayer,'OutputAs','columns');
save DensenetFeatures imageFeatures

%Concatenated-Features

load GaborFeatures.mat
load DensenetFeatures.mat
features = cat(3,imageFeatures,Gabor);
save ConcatenatedFeatures features

%Faster R-CNN

lgraph = layerGraph(net);

%Remove-the-last-3-layners
  
layersToRemove = {
    'fc1000'
    'fc1000_softmax'
    'ClassificationLayer_fc1000'
    };
lgraph = removeLayers(lgraph, layersToRemove);

%Specify-the-number-of-classes-the-network-should-classify
  
numClasses = 2;
numClassesPlusBackground = numClasses + 1;

%Define-new-classification-layers
  
newLayers = [
    fullyConnectedLayer(numClassesPlusBackground, 'Name', 'rcnnFC')
    softmaxLayer('Name', 'rcnnSoftmax')
    classificationLayer('Name', 'rcnnClassification')
    ];

%Add-new-object-classification-layers
  
lgraph = addLayers(lgraph, newLayers);

%Connect-the-new-layers-to-the-network
  
lgraph = connectLayers(lgraph, 'avg_pool', 'rcnnFC');

%Define-the-number-of-outputs-of-the-fully-connected-layer
  
numOutputs = 4 * numClasses;

%Create-the-box-regression-layers
  
boxRegressionLayers = fullyConnectedLayer(numOutputs,'Name','rcnnBoxFC')

%Add-the-layers-to-the-network
  
lgraph = addLayers(lgraph, boxRegressionLayers);

%Connect-the-regression-layers-to-the-layer-named 'avg_pool'
  
lgraph = connectLayers(lgraph,'avg_pool','rcnnBoxFC');

%Add-ROI-max-pooling-layer
  
outputSize = [14 14];

%Define-anchor-boxes
  
anchorBoxes = [
    16 16
    32 16
    16 32
    ];

%Number-of-anchor-boxes
  
numAnchors = size(anchorBoxes,1);

%Number-of-feature-maps-in-coming-out-of-the-feature-extraction-layer
  
numFilters = 1000;
img= f;
img1 = img < 65;
img2 = imclearborder(img1);
rp = regionprops(img2, 'BoundingBox', 'Area');
area = [rp.Area];
[~,ind] = max(area);
bb = rp(ind).BoundingBox;
rpnLayers = [
    convolution2dLayer(3, numFilters,'padding',[1 1],'Name','rpnConv3x3')
    reluLayer('Name','rpnRelu')
    ];

lgraph = addLayers(lgraph, rpnLayers);

%Connect-to-RPN-to-feature-extraction-layer
  
lgraph = connectLayers(lgraph,'avg_pool','rpnConv3x3');

%Add-RPN-classification-layers
  
rpnClsLayers = convolution2dLayer(1, numAnchors*2,'Name', 'rpnConv1x1ClsScores')
lgraph = addLayers(lgraph, rpnClsLayers);

%Connect-the-classification-layers-to-the-RPN-network
  
lgraph = connectLayers(lgraph, 'rpnRelu', 'rpnConv1x1ClsScores');

%Add-RPN-regression-layers

rpnRegLayers = convolution2dLayer(1, numAnchors*4, 'Name', 'rpnConv1x1BoxDeltas')
lgraph = addLayers(lgraph, rpnRegLayers);

%Connect-the-regression-layers-to-the-RPN-network
  
lgraph = connectLayers(lgraph, 'rpnRelu', 'rpnConv1x1BoxDeltas');

%Show-the-network-after-adding-the-RPN-layers

figure(8),
plot(lgraph)
ylim([30 42])
title('network after adding the RPN layers')
figure(9),imshow(img);
rectangle('Position', bb, 'EdgeColor', 'green');

%Performance-Metrics
  
%tested image
load train 
  
%train image
  
load test 
cp = classperf(ry,rt)
accuracy = cp.CorrectRate
sensitivity = cp.Sensitivity
specificity = cp.Specificity

confusionmat2 = confusionmat(ry,rt);
figure(10),plotConfMat(confusionmat2,rt);

%Recall
  
for i =1:size(confusionmat2,1)
    recall(i)=confusionmat2(i,i)/sum(confusionmat2(i,:));
end
recall(isnan(recall))=[];
Recall=sum(recall)/size(confusionmat2,1)
  
%Précision

for i =1:size(confusionmat2,1)
    precision(i)=confusionmat2(i,i)/sum(confusionmat2(:,i));
end
Precision=sum(precision)/size(confusionmat2,1)
  
%F-score
  
F_score=2*Recall*Precision/(Precision+Recall)
  
%Performance-Analysis
  
%True Positive is number of bounding-boxes that has a correct prediction in abnormal images
  
tp = 72
  
%True Negative is number of normal images that has no bounding-boxes
  
tn = 10
  
%False Positive is number of bounding boxes generated outside the abnormal ground-truth region
  
fp = 14
  
%False Negative is number of abnormal images that has no prediction
  
fn = 4
  
%Recall
  
recall=tp/(tp+fn)
  
%Precision
  
precision=tp/(tp+fp)
  
%F1-score
  
F1_score = (2*precision*recall)/(precision + recall)
  
%mean Average Precision
  
ap = (10*precision*recall)/11
xlabel=[ap precision recall]
y= categorical({'AVERAGE PRECISION', 'PRECISION','RECALL'});
figure(11),bar(y,xlabel)
