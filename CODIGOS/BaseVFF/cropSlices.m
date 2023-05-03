
clc
clear all
basePath =  'H:\My Drive\Doutorado\BaseColuna\BaseMatlab8bits\';
resolutions(10000,2) = 0;
contador = 1;
saveDir = 'E:\Datasets\DatasetBalanced\';
% saveDir = 'G:\My Drive\Doutorado\BaseColuna\shared\BaseExanpadida\'


load('H:\My Drive\Doutorado\BaseColuna\fraturados.mat')
load('H:\My Drive\Doutorado\BaseColuna\classes-info.mat')
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

for tw=2:2
    for i=63:64
        for j=1:5
            if (fraturados(i,j) == 0 && a(i,1) > 1)
                fileName = strcat(basePath, num2str(tw, "%.2d"), '-', ...
                    num2str(i, "%.2d"), '-', num2str(j, "%.2d"), '.mat');
                    classe = (a(i,1) == 3 || a(i,1) == 5);
                    clear ROINormalized ROIOriginal ExamNormalized ExamOriginal ExamInfoOriginal pos middle
                    load(fileName);
                   
                    pos = find(sum(sum(ROINormalized>0,3),2) > 0);
                    if (numel(pos) < 6)
                       disp('not enough slices');
                       break;
                    end
%                     [i j numel(pos)]
%                     if (numel(pos) < mymin)
%                         mymin = numel(pos)
%                     end
                        
%                     numSlices(64,5) = numel(pos);
%                     continue
%                     ExamNormalized = permute(ExamNormalized, [1 3 2]);
%                     ROINormalized  = permute(ROINormalized, [1 3 2]);
%                 return
                   middle = round((numel(pos))/2);
%                    middle-2:middle+3

                    
           
                   
%                    numSlices(i,j) = numel(middle-2:middle+3);
                    
                    for z=middle-2:middle+3
                        [tw i j z]
                           clear I GT I2 stats boundingBox GTCropped ICropped ss ICropped2 BW2 mymin mymax stats boundingBox
                            
%                            if (i == 34 || i == 55 || i == 57 || i == 58 || i == 59 || i == 63)
%                                rotation = 90;
%                            else
%                                rotation = 180;
%                            end
%                  
%                             
                          I = squeeze(ExamNormalized(pos(z), : ,:)); % imrotate(squeeze(ExamNormalized(pos(z), : ,:)), rotation);
                          GT = imfill(squeeze(ROINormalized(pos(z), : ,:) > 0), 'holes'); %imrotate(squeeze(ROINormalized(pos(z), : ,:)), rotation);
                          BW2 = imerode(GT, strel('square', 3) );
                          GT = BW2;
                          
                     
%                           cla
%                           imshow(I, [])
%                           return
                          
                          
                          mymin = min(I(I > 0));
                          mymax = max(I(:));
                          mean(I(:))
                          I = I - (mymin + 1);
                          I = I * (255/(mymax - (mymin+1)));
                          I2 =  I;
                          
%                           max(I(:))
%                           imshow(I, [])
%                           return
%                           figure, imshow(I, [])
                          I2(GT == 0) = 0;
                          
                        
%                           mymin = min(I2(GT == 1));
%                           mymax = max(I2(GT == 1));
                          
                        
                          clear B L
%                           [B,L] = bwboundaries(GT == 1,'noholes');
%                           cla
%                           subplot(1,2,1), 
%                           imshow(I2, []), 
%                           hold on
%                           
%                           for k = 1:length(B)
%                                boundary = B{k};
%                                plot(boundary(:,2), boundary(:,1), 'r', 'LineWidth', 2)
%                           end
%                           subplot(1,2,2), imshow(GT, []),
%                           return
%                           pause(0.1)
%                           a(i)
%                           return
                            
                          stats = regionprops('table',GT,'Centroid','BoundingBox', 'Image', 'Orientation');
                          boundingBox = round(double(stats.BoundingBox));
                          
%                           length(stats.Image)
%                           stats.Image
                          
%                           GTCropped = 0;
%                           
%                           if ( length(stats.Image) > 1)
%                               for idx=1:length(stats.Image)
%                                   idx
%                                   temp = stats{idx,1}
% %                                   figure, imshow(stats{idx,1}.Image, [])
% %                                 GTCropped = GTCropped + cell2mat(stats(idx).Image)>0;
%                               end
%                           else
%                               
%                           end
                          
                          GTCropped = cell2mat( stats.Image ) > 0;
                          
                          ICropped = I2(boundingBox(2)-2:boundingBox(2)+boundingBox(4)+2,boundingBox(1)-2:boundingBox(1)+boundingBox(3)+2);
                        
%                           [size(GTCropped); size(ICropped)]
%                           ss = [maxs - size(ICropped)];
%                           if (ss(1) < 1 || ss(2) < 1)
%                               disp('error')
%                               continue;
%                           end
%                           ICropped2(maxs(1), maxs(2)) = uint8(0);
%                           ICropped2(round(ss/2):round(ss/2)+size(ICropped,1)-1, round(ss/2):round(ss/2)+size(ICropped,2)-1) = ICropped;
%                           ICropped = histeq(ICropped);
%                           ICropped(GTCropped) = 0;
%                             ICropped = ((ICropped - mymin)/(mymax-mymin));
%                           imshow(ICropped, [])
%                           return

                          resolutions(contador, :) = [ICropped(3) ICropped(4)];
                          contador = contador + 1;
                          
                          mymin = min(ICropped(ICropped > 0))
                          mymax = max(ICropped(:))
                          
                          
%                           imshow(ICropped, [])
%                           return
                          ICropped = ICropped - (mymin + 1);
                          ICropped = ICropped * (255/(mymax-(mymin + 1)));
                           mymin = min(ICropped(ICropped > 0))
                          mymax = max(ICropped(:))
                          imshow(ICropped)
                            return
                 
                          
%                           imshow(ICropped);
%                           return
                          
                           roiName = strcat('T', num2str(tw, "%.2d"), '-E', num2str(i, "%.2d"), '-L', num2str(j, "%.2d"), '-S', num2str(pos(z), "%.2d"), '.bmp');
                           roiNameMask = strcat('T', num2str(tw, "%.2d"), '-E', num2str(i, "%.2d"), '-L', num2str(j, "%.2d"), '-S', num2str(pos(z), "%.2d"), '.bmp');
                           roiName = strcat(saveDir,'Class-', num2str(uint8(classe), "%.1d"),'\', roiName);
                           roiName2 = strcat(saveDir,'Class-', num2str(uint8(classe), "%.1d"),'\masks\', roiNameMask);
                           ID{cont,1} = strcat('T', num2str(tw, "%.2d"), '-E', num2str(i, "%.2d"), '-L', num2str(j, "%.2d"), '-S', num2str(pos(z), "%.2d"));
                           Image{cont,1} = roiName;
                           Mask{cont,1} = roiName2;
                           Label(cont,1) = uint8(classe);
                           cont = cont+1;
%                            
%                            imshow(ICropped)
                           imwrite(ICropped, roiName)
%                            return
                           ROI  = logical(ICropped ~= 0);
                           ROI = imfill(ROI > 0, 'holes');
%                             nhdr_nrrd_write(roiNameMask,ROI)
                           imwrite(ROI, roiName2)
%                            numSlices(i,j) = numSlices(i,j) + 1;
                           
                           continue
                         
%                           if (~classe) % 2 = Sem fratura
%                                roiName = strcat(saveDir,'Class-0\T', num2str(tw, "%.2d"), '-E', num2str(i, "%.2d"), '-L', num2str(j, "%.2d"), '-S', num2str(pos(z), "%.2d"), '.bmp');
%                                imwrite(ICropped, roiName)
%                                numSlices(i,j) = numSlices(i,j) + 1;
%                           elseif(classe) % 4 = Fratura
%                                roiName = strcat(saveDir,'Class-1\T', num2str(tw, "%.2d"), '-E', num2str(i, "%.2d"), '-L', num2str(j, "%.2d"), '-S', num2str(pos(z), "%.2d"), '.bmp');
%                                imwrite(ICropped, roiName)
%                                cont2 = cont2 + 1;
%                           else
%                               disp('Somtehing went wrong')
%                           end
                 
                    
%                           if (a(i) == 1) % 1 = Massa óssea normal
%                                imwrite(ICropped2, strcat(saveDir,'1-Healthy\T', num2str(tw, "%.2d"), '-E', num2str(i, "%.2d"), '-L', num2str(j, "%.2d"), '-S', num2str(z, "%.2d"), '.bmp'))
%                           elseif (a(i) == 2 ) % 2 = Osteopenia sem fratura
%                                imwrite(ICropped2, strcat(saveDir,'2-OsteopeniaSemFratura\T', num2str(tw, "%.2d"), '-E', num2str(i, "%.2d"), '-L', num2str(j, "%.2d"), '-S', num2str(z, "%.2d"), '.bmp'))
%                           elseif(a(i) == 4) % 4 = Osteoporose sem Fratura
%                                imwrite(ICropped2, strcat(saveDir,'4-OsteoporoseSemFratura\T', num2str(tw, "%.2d"), '-E', num2str(i, "%.2d"), '-L', num2str(j, "%.2d"), '-S', num2str(z, "%.2d"), '.bmp'))
%                           elseif (a(i) == 3) % 3 = Osteopenia com fratura,
%                                imwrite(ICropped2, strcat(saveDir,'3-OsteopeniaComFratura\T', num2str(tw, "%.2d"), '-E', num2str(i, "%.2d"), '-L', num2str(j, "%.2d"), '-S', num2str(z, "%.2d"), '.bmp'));
%                           elseif (a(i) == 5) % 5 = Osteoporose com fratura
%                                imwrite(ICropped2, strcat(saveDir,'5-OsteoporoseComFratura\T', num2str(tw, "%.2d"), '-E', num2str(i, "%.2d"), '-L', num2str(j, "%.2d"), '-S', num2str(z, "%.2d"), '.bmp'));
%                           end
                    end
            end
        end
    end
end

% tb = table(ID, Image, Mask, Label);
% filename = strcat('E:\Datasets\DatasetBalanced\ExtractFeatures.csv');
% writetable(tb,filename)