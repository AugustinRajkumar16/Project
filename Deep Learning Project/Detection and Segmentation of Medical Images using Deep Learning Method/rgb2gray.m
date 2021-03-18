"" we can select the whole image folder from dataset to Convert Gray-scale Image ""
  
%Open folder selection dialog box, for selecting input and output folders
  
indir = uigetdir(cd, 'Select input folder');
outdir = uigetdir(cd, 'Select output folder');
directory = dir([indir, '\', '*.tif']);

for i = 1 : length(directory)
    filename = directory(i).name;
    rgb_img = imread([indir, '\', filename]);    
    if (ndims(rgb_img) == 3) %Make sure img is RGB (not gray).
        img = rgb2gray(rgb_img);
        [~,name,~] = fileparts(filename);
        gsFilename = sprintf('%s_gs.jpg', name);
        %Save gray image to outdir (keep original name).
        imwrite(img, [outdir, '\', gsFilename]);
    end
end



