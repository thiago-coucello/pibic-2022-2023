
clc
clear all

basePath = 'H:\My Drive\Doutorado\BaseColuna\shared\Datasets\DatasetBalanced2\';

img1 = strcat(basePath, 'Class-0/T01-E02-L01-S08.bmp');
img2 = strcat(basePath, 'Class-0/T02-E12-L05-S08.bmp');
img3 = strcat(basePath, 'Class-1/T01-E08-L01-S07.bmp');
img4 = strcat(basePath, 'Class-1/T02-E21-L04-S07.bmp');

I1 = imread(img1);
I2 = imread(img2);
I3 = imread(img3);
I4 = imread(img4);

% 80x80
I1_new(80,80) = uint8(0);
I2_new(80,80) = uint8(0);
I3_new(80,80) = uint8(0);
I4_new(80,80) = uint8(0);

[m, n] = size(I1);
I1_new(40-(round(m/2))+1:40+(round(m/2)-1), 40-(round(n/2))+1:40+(round(n/2))-1) = I1;

[m, n] = size(I2);
I2_new(40-(round(m/2))+1:40+(round(m/2)-1), 40-(round(n/2))+1:40+(round(n/2))-1) = I2;

[m, n] = size(I3);
I3_new(40-(round(m/2))+1:40+(round(m/2)-1), 40-(round(n/2))+1:40+(round(n/2))) = I3;

[m, n] = size(I4);
I4_new(40-(round(m/2))+1:40+(round(m/2)-1), 40-(round(n/2))+1:40+(round(n/2))-1) = I4;


% 128x128
I1_new = imresize(I1_new, [128,128]);
subplot(1,4,1), imshow(I1_new,[])
subplot(1,4,2), imshow(I2_new,[])
subplot(1,4,3), imshow(I3_new, [])
subplot(1,4,4), imshow(I4_new, [])

imwrite(I1_new, strcat('C:\OsteoporosisAnalysis\T01-E02-L01-S08.jpg'))
imwrite(I2_new, strcat('C:\OsteoporosisAnalysis\T02-E12-L05-S08.jpg'))
imwrite(I3_new, strcat('C:\OsteoporosisAnalysis\T01-E08-L01-S07.jpg'))
imwrite(I4_new, strcat('C:\OsteoporosisAnalysis\T02-E21-L04-S07.jpg'))

