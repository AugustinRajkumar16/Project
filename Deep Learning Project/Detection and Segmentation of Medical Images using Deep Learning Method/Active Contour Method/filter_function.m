function [smth]=filter_function(image,sigma)

smask= fspecial('gaussian',3,sigma);
smth=filter2(smask,image,'full');
