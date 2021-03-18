function features = GaborFeatures(img,gabor,d1,d2)
%GABORFEATURES Summary of this function goes here
if(nargin~=4)
    error('Incorrect number')
end

if size(img,3)==3
    warning('RGB to Greyscale')
    img=rgb2gray(img);    
end

img=double(img);
%The input image by each GaborFilter
[scales,orientations]=size(gabor);
gaborResult=cell(scales,orientations);
for i=1:scales
    for j=1:orientations
        gaborResult{i,j}=imfilter(img,gabor{i,j});
    end
end

%Creating Feature
features=[];
for i=1:scales
    for j=1:orientations
        gaborAbs=abs(gaborResult{i,j});
        gaborAbs=downsample(gaborAbs,d1);
        gaborAbs=downsample(gaborAbs.',d2);
        gaborAbs=gaborAbs(:);
        gaborAbs=(gaborAbs-mean(gaborAbs))/std(gaborAbs,1);
        features=[features; gaborAbs];
    end
end

end

