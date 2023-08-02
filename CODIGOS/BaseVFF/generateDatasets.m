
clc
clear all
basePath =  'C:\PIBIC\2022-2023\CODIGOS\BaseVFF\BaseMatlab8bits\';
resolutions(10000,2) = 0;
contador = 1;
saveDir = 'C:\PIBIC\2022-2023\Datasets';
% saveDir = 'G:\My Drive\Doutorado\BaseColuna\shared\BaseExanpadida\'

if ~exist(saveDir, "dir")
    mkdir(saveDir);
else
    rmdir(saveDir, "s");
    mkdir(saveDir);
end

load('C:\PIBIC\2022-2023\CODIGOS\BaseVFF\fraturados.mat')
load('C:\PIBIC\2022-2023\CODIGOS\BaseVFF\classes-info.mat')
% return
numSlices(64,5) = 0;

maxs = [80  80];
mymin = 9999;

cont1 = 0;
cont2 = 0;
ID{2400,1} = '';
Image{2400,1} = '';
Mask{2400,1} = '';
Label(2400,1) = 0;
cont = 1;

% 4-2, 6-2, 8-2 3 4,

for tw=1:2
    for i=1:64
        for j=1:5
            if (fraturados(i,j) == 0 && a(i,1) > 1)
                fileName = strcat(basePath, num2str(tw, "%.2d"), '-', ...
                    num2str(i, "%.2d"), '-', num2str(j, "%.2d"), '.mat');
                classe = (a(i,1) == 3 || a(i,1) == 5);
                clear ROINormalized ROIOriginal ExamNormalized ExamOriginal ExamInfoOriginal pos middle
                load(fileName);
                
                pos = find(sum(sum(ROINormalized>0,3),2) > 0);
                middle = round((numel(pos))/2);
                
                for z=middle-2:middle+3
                    [tw i j z]
                    clear I GT I2 stats boundingBox GTCropped ICropped ss ICropped2 BW2 mymin mymax stats boundingBox
                    
                    I = squeeze(ExamNormalized(pos(z), : ,:)); % imrotate(squeeze(ExamNormalized(pos(z), : ,:)), rotation);
                    
                    GT = imfill(squeeze(ROINormalized(pos(z), : ,:) > 0), 'holes'); %imrotate(squeeze(ROINormalized(pos(z), : ,:)), rotation);
                    BW2 = imerode(GT, strel('square', 3) );
                    GT = BW2;
                    
                    
                    mymin = min(I(I > 0));
                    mymax = max(I(:));
                    mean(I(:));
                    I = I - (mymin + 1);
                    I = I * (255/(mymax - (mymin+1)));
                    I2 =  I;
                    
                    I2(GT == 0) = 0;
                    
                    stats = regionprops('table',GT,'Centroid','BoundingBox', 'Image', 'Orientation');
                    boundingBox = round(double(stats.BoundingBox));
                    
                    
                    GTCropped = cell2mat( stats.Image ) > 0;
                    
                    ICropped = I2(boundingBox(2)-2:boundingBox(2)+boundingBox(4)+2,boundingBox(1)-2:boundingBox(1)+boundingBox(3)+2);
                    
                    resolutions(contador, :) = [ICropped(3) ICropped(4)];
                    contador = contador + 1;
                    
                    mymin = min(ICropped(ICropped > 0));
                    mymax = max(ICropped(:));
                    
                    ICropped = ICropped - (mymin + 1);
                    ICropped = ICropped * (255/(mymax-(mymin + 1)));
                    mymin = min(ICropped(ICropped > 0));
                    mymax = max(ICropped(:));
                    
                    roiName = strcat('T', num2str(tw, "%.2d"), '-E', num2str(i, "%.2d"), '-L', num2str(j, "%.2d"), '-S', num2str(pos(z), "%.2d"), '.tif');
                    
                    ROI  = logical(ICropped ~= 0);
                    ROI = imfill(ROI > 0, 'holes');
                    cropAndSaveImages(ICropped, ROI, saveDir, roiName, classe);
                    
                end
            end
        end
    end
end

function cropAndSaveImages(Image, Mask, saveDir, roiName, label)
percentages = 0:0.05:0.95;
for i = 1:length(percentages)
    close all;
    subset = num2str(uint8(100 * (1 - percentages(i))), "%02d");
    subsetDir = strcat(saveDir, "\", subset);
    imageDir = strcat(subsetDir, "\", 'Class-', num2str(uint8(label), "%.1d"), "\");
    maskDir = strcat(subsetDir, "\", 'Class-', num2str(uint8(label), "%.1d"), "\masks\");
    
    imageFile = strcat(imageDir, roiName);
    maskFile = strcat(maskDir, roiName);
    ignoredFile = strcat(subsetDir, "\ignored.txt");
    
    if ~exist(subsetDir, "dir")
        mkdir(subsetDir);
    end
    
    if ~exist(imageDir, "dir")
        mkdir(imageDir);
    end
    
    if ~exist(maskDir, "dir")
        mkdir(maskDir);
    end
    
    totalImage = sum(Image ~= 0, "all");
    target = 1 - percentages(i);
    croppedMask = Mask;
    verificationMask = Mask;
    ratio = 1;
    threshold = 1e-2;
    relativeError = abs(target - ratio) / abs(target);
    
    while relativeError > threshold
        boundaries = bwboundaries(croppedMask); % Pega contornos da máscara
        boundary = boundaries{1};   % Seleciona o primeiro (único)
        
        % Remove os pontos da borda da máscara utilizada para verificação
        for k = 1:length(boundary)
            x = boundary(k, 1);
            y = boundary(k, 2);
            verificationMask(x, y) = 0;
        end
        
        % Calcula a razão mascara/original para verificar se passou do
        % valor alvo
        totalMask = sum(verificationMask, "all");
        ratio = totalMask / totalImage;
        relativeError = abs(target - ratio) / abs(target);
        
        % Se a razão for menor que o valor alvo significa que removeu
        % demais
        if (ratio < target)
            % Calcula a quantidade de pontos a ser removidos para alcançar
            % o alvo e os remove
            pointsToRemove = floor((1 - target) * totalImage) - (totalImage - sum(croppedMask, "all"));
            for k = 1:pointsToRemove
                x = boundary(k, 1);
                y = boundary(k, 2);
                croppedMask(x, y) = 0;
            end
            
            totalCroppedMask = sum(croppedMask, "all");
            ratioCroppedMask = totalCroppedMask / totalImage;
            break;
            % A razão está maior que o valor alvo, então a borda que foi
            % excluída deve ser excluída por completo
        else
            croppedMask = verificationMask;
        end
    end
    
    croppedImage = uint8(zeros(size(Image)));
    croppedImage(croppedMask) = Image(croppedMask);
    
    
    imageBoundaries = bwboundaries(Image, "noholes");
    imageBoundary = imageBoundaries{1};
    
    croppedImageBoundaries = bwboundaries(croppedImage, "noholes");
    croppedImageBoundary = croppedImageBoundaries{1};
    
    if (isequaln(subset, "90") || isequaln(subset, "60") || isequaln(subset, "30") || isequaln(subset, "10"))
        
        plotImage = Image;
        plotImage(~Mask) = 255;
        
        figure('units','normalized','outerposition',[0 0 1 1]),
        imshow(plotImage);
        hold on;
        plot(imageBoundary(:, 2), imageBoundary(:, 1), 'r');
        plot(croppedImageBoundary(:, 2), croppedImageBoundary(:, 1), 'c');
        % legend("Contorno original", "Contorno reduzido");
        hold off;
        
        %{
        subplot(1, 3, 2); imshow(croppedImage); title(strcat("Imagem Reduzida - ", subset));
        hold on;
        plot(imageBoundary(:, 2), imageBoundary(:, 1), 'c');
        plot(croppedImageBoundary(:, 2), croppedImageBoundary(:, 1), 'r');
        legend("Contorno original", "Contorno reduzido");
        hold off;

        subplot(1, 3, 3); imshow(croppedMask); title(strcat("Máscara Reduzida - ", subset));
        hold on;
        plot(imageBoundary(:, 2), imageBoundary(:, 1), 'c');
        plot(croppedImageBoundary(:, 2), croppedImageBoundary(:, 1), 'r');
        legend("Contorno original", "Contorno reduzido", "Location", "northeast");
        hold off;
        %}
    end
    
    if all(croppedImage == 0, "all") || all(croppedMask == 0, "all")
        arq = fopen(ignoredFile, "a");
        fprintf(arq, "%s\n", roiName);
        fclose(arq);
    else
        imwrite(croppedImage, imageFile);
        imwrite(croppedMask, maskFile);
    end
end
end

% tb = table(ID, Image, Mask, Label);
% filename = strcat('E:\Datasets\DatasetBalanced\ExtractFeatures.csv');
% writetable(tb,filename)
