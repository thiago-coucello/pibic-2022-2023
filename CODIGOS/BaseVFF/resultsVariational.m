clc
clear all

csvName = 'H:\My Drive\Doutorado\BaseColuna\shared\Datasets\DatasetBalanced2\Results\csvs\InceptionV3.csv';

mytable = readtable(csvName);

resnet = mytable(strcmp(mytable.network, 'InceptionV3') & mytable.folder < 10, :);
freezePercentage{1} = '';
cont = 1;
for i=128:128:1024
    for j=0.1:0.1:0.5
        for k = 0.1:0.1:0.5
             
%             double(resnet.DropOut)
            clear idx1 idx2 idx3  idx
            idx1 = resnet.DenseNum == i;
            idx2 = floor(100*resnet.DropOut) == floor(100*j);
            idx3 = floor(100*resnet.FreezePercentage) == floor(100*k);
            idx = idx1 & idx2 & idx3;
            idxs = [idx1, idx2, idx3];
            freezes = resnet( idx  , :);
            if size(freezes, 1) > 2
               [cont i 10*j 10*k]
                freezePercentage{cont} = freezes;
                cont = cont + 1;
%             else
%                 [i j k]
            end
        end
        
    end
end

data(5, size(freezePercentage, 2)) = 0;
for i=1:size(freezePercentage, 2)
    data(1:9, i) = freezePercentage{i}.val_accuracy(1:9,:);
end

[a b] = max(mean(data))

plot(mean(data))