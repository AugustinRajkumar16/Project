

clc;
clear all;
close all;

[file,path]=uigetfile('*.jpg','select a input image');
str=strcat(path,file);
I=imread(str);
figure(1),imshow(I),title('Input Image');

% sigma_val = str2double(1.00);
f =filter_function(I,1.00);
figure(2),imshow(f,[]),title('Filter Image');
[xs,ys]=getsnake(f);
figure(3),imshow(f,[]);

val = iteration(f,xs,ys,0.40,0.20,1.00,0.15,0.30,0.40,0.70,300);
