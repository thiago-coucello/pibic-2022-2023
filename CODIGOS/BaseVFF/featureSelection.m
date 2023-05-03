clc
clear all

baseSavePath = 'E:\Datasets\DatasetBalanced\Features\';
csvFeatures = readtable('E:\Datasets\DatasetBalanced\Features\Features.csv');
csvFeatures.Image = [];
csvFeatures.Mask = [];
csvFeatures.Reader = [];
csvFeatures = sortrows(csvFeatures, 'ID');

% for i=1:length(csvFeatures.Properties.VariableNames)
%     char(csvFeatures.Properties.VariableNames(i))
% end

% return
labels = csvFeatures.Label;
% labels = strcmp(labels, 'ComFratura');

featuresAll = csvFeatures(1:end, 2:end);
[idx,scores] = fscchi2(featuresAll,'Label');
normalizedScores = scores/max(scores(:));

% bar(sort(normalizedScores))
% xlabel('Feature rank')
% ylabel('Feature importance score')

featuresAll2 = featuresAll;
thresh = 0.15;
% sum(normalizedScores > 0.22)
featuresAll2(:, normalizedScores < thresh) = [];

% return
% featuresAll = featuresAll{:, :};

% return
% classes = strcmp( T{:,end}, 'ComFratura');
% features = T{:,2:end-1};  
% OriginalNames = T{:,1};
% % classes = ;
% features = normalizeData(features);
% features = [features classes];


partitionPath = 'E:\Datasets\DatasetBalanced\Partitions\';

% a = csvFeatures.image;


AllResults{100} = 0;
cont = 1;

%  Trains( size( features, 1), 100 ) = 0;
%  Tests( size( features, 1), 100 ) = 0;
for idx=1:100
    [idx thresh] 
    clear filename csv i j Trainning Test  
%   SAME TEST AND TRAIN AS THE ONES USED IN DEEP LEARNING
    filename = strcat(partitionPath, num2str(idx,'%1d'), '.csv');
    
    
% 
    csv = readtable(filename);
    
    for i=1:size(csv,1)
        ID = char(csv.Image(i));
        ID = ID(end-18:end-4);
        csv.Image(i) = cellstr(ID);
    end
    
    csv = sortrows(csv, 'Image');
    labels = csv.Class;
    
    if (sum(strcmpi(csv.Image, csvFeatures.ID)) == size(csv,1))
        
        Trainning = csv.Train;
        Test = csv.Test;
        clear idx1 idx2 idx3 idx4
        idx1 = Trainning == 1;
        idx3 = Test == 1;
        clear featuresTestClassFractures featuresTestClassNoFractures featuresTrainClassFractures featuresTrainClassNoFractures
        featuresTrainClassFractures = featuresAll2(idx1, 1:end);
        labelsTrain = labels(idx1);
        featuresTrainClassFractures.Label = labelsTrain;
        featuresTestClassFractures = featuresAll2(idx3, 1:end);
        labelsTest = labels(idx3);
        featuresTestClassFractures.Label = labelsTest;
        clear fname1 fname2 fname3 fname4
        fname1 = strcat(baseSavePath, num2str(idx,'%.2d'), '_', num2str(100*thresh,'%.3d'), '_train.csv');
        fname3 = strcat(baseSavePath, num2str(idx,'%.2d'), '_', num2str(100*thresh,'%.3d'), '_test.csv');
        writetable(featuresTrainClassFractures, fname1, 'delimiter', ',');
        writetable(featuresTestClassFractures, fname3, 'delimiter', ',');
        continue
        
    else
        idx
        disp('error');
        break
    end
   
   
    
    
%     idx2 = Trainning == 1 & labels == 0;
    
    
%     idx4 = Test == 1 & labels == 0;
    
    
    
%     featuresTrainClassNoFractures = featuresAll2(idx1, 1:end-1);
    
    
%     featuresTestClassNoFractures = featuresAll2(idx4, 1:end-1);

    
    clear fname1 fname2 fname3 fname4
    fname1 = strcat(baseSavePath, num2str(idx,'%.2d'), '-train.csv');
%     fname2 = strcat(baseSavePath, num2str(idx,'%.2d'), '-train-nofractures.csv');
    
%     fname4 = strcat(baseSavePath, num2str(idx,'%.2d'), '-test-nofractures.csv');
    
    
%     writetable(featuresTrainClassNoFractures, fname2, 'delimiter', ',');
    
%     writetable(featuresTestClassNoFractures, fname4, 'delimiter', ',');
    continue
    
%     for i =1:size(csv,1)   
%         for j=1:length( Tests )
%             if (strcmp(string(csv.I
% mage(i)), string( OriginalNames{j}))) 
%                 if (csv.Trainning(i) == 1)
%                     Trains(j,idx) = 1;
%                 elseif (csv.Test(i) == 1)
%                     Tests(j,idx) = 1;
%                 end
%                 break;
%             end
%         end 
%     end
%    save('G:\My Drive\Doutorado\BaseColuna\shared\Jonathan\TrainTest.mat', 'Trains', 'Tests', 'OriginalNames')
% end

return

for i=0.22:0.01:0.22
    
    clear sfeatures numFeatures myNames names names2 inputTable predictors isCategoricalPredictor r
    [sfeatures, escolhidos, scores] = selectFeatures(features, i);
    
  
     
%     h = histogram(scores,10) ;
% x = h.BinEdges ;
% y = h.Values ;
% text(x(1:end-1),y,num2str(y'),'vert','bottom','horiz','center'); 
% box off
%     
 

%     numFeatures = size(sfeatures,2)-1;
%    
%     myNames = [1:numFeatures];
%     names = cellstr(num2str(myNames'))';
%     names2 = names;
%     names{end+1} = 'Y';
%     inputTable = array2table(sfeatures, 'VariableNames', names);
%     predictors = inputTable(:, names2);
%     isCategoricalPredictor(numFeatures) = false;




    r{6} = 0;
    clear results
    [results] = trainClassifierEnsemble(sfeatures, Trains(:,idx), Tests(:,idx));
    r{1} = results;

    clear results
    [results] = trainClassifierKnn(sfeatures, Trains(:,idx), Tests(:,idx));
    r{2} = results;

    clear results
    [results] = trainClassifierSvm(sfeatures, Trains(:,idx), Tests(:,idx));
    r{3} = results;

    clear results
    [results] = trainClassifierTree(sfeatures, Trains(:,idx), Tests(:,idx));
    r{4} = results;

    clear results
    [results] = trainClassifierDiscriminant(sfeatures, Trains(:,idx), Tests(:,idx));
    r{5} = results;
    
    clear results
    [results] = trainClassifierNaiveBayes(sfeatures, Trains(:,idx), Tests(:,idx));
    r{6} = results;
    
%         return
    
%       return

    AllResults{cont} = r;
    cont = cont + 1;
    save('G:\My Drive\Doutorado\BaseColuna\shared\Jonathan\Results\TestTrainResults22.mat', 'AllResults')
end

end

% plot(fpr,tpr)
% xlabel('False positive rate')
% ylabel('True positive rate')
% title('ROC Curve')



