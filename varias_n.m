close all; clc; clear
pastan   = 'C:\Users\vinic\Desktop\play8_rail2_r\';
pastaout = [pastan, 'nao2\'];
mkdir(pastaout)
arquivo  = [pastan, 'frame0438.jpg'];

im = imread(arquivo);
imsize = size(im);
window_base = [450 200];

count = 0;
for s=1:2
    scale_base   = floor(window_base/s);
    
    many_steps = imsize(1:2)./scale_base;
    
    step_size  = imsize(1:2)./many_steps;
    
    for x=1:step_size(1):imsize(1)-step_size(1)-1
        for y=1:step_size(2):imsize(2)-step_size(2)-1
            crop = im(x:x+step_size(1), y:y+step_size(2), 1:imsize(3));
%             imshow(crop)
            imwrite(crop, [pastaout, 'h', num2str(count), '.jpg'])
            count = count+1;
        end
    end
end