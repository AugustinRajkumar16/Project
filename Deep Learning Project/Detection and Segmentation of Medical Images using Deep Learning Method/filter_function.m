function [smth]=filter_function(image,sigma)
% c=ceil(3*sigma)
% noise=fspecial('gaussian');
% smth=filter2(image,noise);

% smth = imgaussfilt(image,sigma);
smask= fspecial('gaussian',3,sigma);
smth=filter2(smask,image,'full');