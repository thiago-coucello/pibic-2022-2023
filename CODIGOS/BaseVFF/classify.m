

function [results] = classify(inputTable, predictors, isCategoricalPredictor, option)


response = inputTable.Y;

switch option
    case 'ensemble' % ----------------------------------------------
        disp('Ensemble Classification')
        
        % Train a classifier
% This code specifies all the classifier options and trains the classifier.
template = templateTree(...
    'MaxNumSplits', 30, ...
    'NumVariablesToSample', 'all');
classificationEnsemble = fitcensemble(...
    predictors, ...
    response, ...
    'Method', 'GentleBoost', ...
    'NumLearningCycles', 143, ...
    'Learners', template, ...
    'LearnRate', 0.9140957144341983, ...
    'ClassNames', [0; 1]);
        
        
        trainedClassifier.ClassificationEnsemble = classificationEnsemble;
       rng('default') % For reproducibility 
        % Create the result struct with predict function
        cvp = cvpartition(response, 'Holdout', 0.2);
        trainingPredictors = predictors(cvp.training, :);
trainingResponse = response(cvp.training, :);
predictorExtractionFcn = @(t) t(:, predictorNames);

ensemblePredictFcn = @(x) predict(classificationEnsemble, x);
validationPredictFcn = @(x) ensemblePredictFcn(x);

% ensemblePredictFcn = @(x) predict(cEnsemble, x);
% trainedClassifier.predictFcn = @(x) ensemblePredictFcn(predictorExtractionFcn(x));
        
%         trainInds = training(cvp);
% sampleInds = test(cvp);
% trainingData = meas(trainInds,:);
% sampleData = meas(sampleInds,:);

validationPredictors = predictors(cvp.test, :);
validationResponse = response(cvp.test, :);

class = classify(cvp.training,trainingResponse,trainingPredictors);
cm = confusionchart(validationPredictors,class);

%         partitionedModel = crossval(trainedClassifier.ClassificationEnsemble, 'KFold', 10, 'nprint', 1);
%         
%         [validationPredictions, validationScores] = kfoldPredict(partitionedModel);
%         [~,posterior] = kfoldPredict(partitionedModel);
%         [fpr,tpr,~,auc] = perfcurve(inputTable.Y, posterior(:,2),partitionedModel.ClassNames(2));
%         
%         
%         C = confusionmat(response, validationPredictions);
%         [val1, val2] = calculateMeasures(C, auc);
        
        
    case 'knn' % ----------------------------------------------
        disp('KNN Classification') 
       
        % Train a classifier
% This code specifies all the classifier options and trains the classifier.
classificationKNN = fitcknn(...
    predictors, ...
    response, ...
    'Distance', 'Spearman', ...
    'Exponent', [], ...
    'NumNeighbors', 7, ...
    'DistanceWeight', 'SquaredInverse', ...
    'Standardize', true, ...
    'ClassNames', [0; 1]);
        
        predictorExtractionFcn = @(x) array2table(x, 'VariableNames', predictorNames);
        knnPredictFcn = @(x) predict(classificationKNN, x);
        trainedClassifier.predictFcn = @(x) knnPredictFcn(predictorExtractionFcn(x));
        
        trainedClassifier.ClassificationKNN = classificationKNN;
        partitionedModel = crossval(trainedClassifier.ClassificationKNN, 'KFold', 10);
        
        [validationPredictions, validationScores] = kfoldPredict(partitionedModel);
        [~,posterior] = kfoldPredict(partitionedModel);
        [fpr,tpr,~,auc] = perfcurve(inputTable.Y, posterior(:,2),partitionedModel.ClassNames(2));
        
        C = confusionmat(response, validationPredictions);
        [val1, val2] = calculateMeasures(C, auc);
        
    case 'svm' % --------------------------------------------------------
        disp('Support Vector Machine Classification')
        
classificationSVM = fitcsvm(...
    predictors, ...
    response, ...
    'KernelFunction', 'polynomial', ...
    'PolynomialOrder', 2, ...
    'KernelScale', 1, ...
    'BoxConstraint', 0.001002975245133279, ...
    'Standardize', true, ...
    'ClassNames', [0; 1]);
        
       % Create the result struct with predict function
predictorExtractionFcn = @(x) array2table(x, 'VariableNames', predictorNames);
svmPredictFcn = @(x) predict(classificationSVM, x);
trainedClassifier.predictFcn = @(x) svmPredictFcn(predictorExtractionFcn(x));

% Add additional fields to the result struct
trainedClassifier.ClassificationSVM = classificationSVM;
        
        partitionedModel = crossval(trainedClassifier.ClassificationSVM, 'KFold', 10);
        
        [validationPredictions, validationScores] = kfoldPredict(partitionedModel);
        [~,posterior] = kfoldPredict(partitionedModel);
        [fpr,tpr,~,auc] = perfcurve(inputTable.Y, posterior(:,2),partitionedModel.ClassNames(2));
        
        C = confusionmat(response, validationPredictions);
        [val1, val2] = calculateMeasures(C, auc);
        
    case 'trees' % ------------------------------------------------
        disp('Decision Tree Classification')
        classificationTree = fitctree(...
    predictors, ...
    response, ...
    'SplitCriterion', 'deviance', ...
    'MaxNumSplits', 54, ...
    'Surrogate', 'off', ...
    'ClassNames', [0; 1]);

        
        predictorExtractionFcn = @(x) array2table(x, 'VariableNames', predictorNames);
        treePredictFcn = @(x) predict(classificationTree, x);
        trainedClassifier.predictFcn = @(x) treePredictFcn(predictorExtractionFcn(x));
        
        trainedClassifier.ClassificationTree = classificationTree;
        partitionedModel = crossval(trainedClassifier.ClassificationTree, 'KFold', 10);
        
        [validationPredictions, validationScores] = kfoldPredict(partitionedModel);
        [~,posterior] = kfoldPredict(partitionedModel);
        [fpr,tpr,~,auc] = perfcurve(inputTable.Y, posterior(:,2),partitionedModel.ClassNames(2));
        
        C = confusionmat(response, validationPredictions);
        [val1, val2] = calculateMeasures(C, auc);
        
    case 'discriminant' % --------------------------------------------
        disp('Discriminant Analysis Classification')
        
   classificationDiscriminant = fitcdiscr(...
    predictors, ...
    response, ...f
    'DiscrimType', 'pseudoLinear', ...
    'Gamma', 0, ...
    'FillCoeffs', 'off', ...
    'ClassNames', [0; 1]);
        
        predictorExtractionFcn = @(x) array2table(x, 'VariableNames', predictorNames);
        discriminantPredictFcn = @(x) predict(classificationDiscriminant, x);
        trainedClassifier.predictFcn = @(x) discriminantPredictFcn(predictorExtractionFcn(x));
        
        trainedClassifier.ClassificationDiscriminant = classificationDiscriminant;
        
        partitionedModel = crossval(trainedClassifier.ClassificationDiscriminant, 'KFold', 10);
        
        [validationPredictions, validationScores] = kfoldPredict(partitionedModel);
        [~,posterior] = kfoldPredict(partitionedModel);
        [fpr,tpr,~,auc] = perfcurve(inputTable.Y, posterior(:,2),partitionedModel.ClassNames(2));
        
        C = confusionmat(response, validationPredictions);
        [val1, val2] = calculateMeasures(C, auc);
        
    case 'bayes' % ------------------------------
        disp('Naive Bayes Classification')
       % Expand the Distribution Names per predictor
        % Numerical predictors are assigned either Gaussian or Kernel distribution and categorical predictors are assigned mvmn distribution
        % Gaussian is replaced with Normal when passing to the fitcnb function
  distributionNames =  repmat({'Kernel'}, 1, length(isCategoricalPredictor));
distributionNames(isCategoricalPredictor) = {'mvmn'};

if any(strcmp(distributionNames,'Kernel'))
    classificationNaiveBayes = fitcnb(...
        predictors, ...
        response, ...
        'Kernel', 'Normal', ...
        'Support', 'Unbounded', ...
        'DistributionNames', distributionNames, ...
        'ClassNames', [0; 1]);
else
    classificationNaiveBayes = fitcnb(...
        predictors, ...
        response, ...
        'DistributionNames', distributionNames, ...
        'ClassNames', [0; 1]);
end
        
        predictorExtractionFcn = @(x) array2table(x, 'VariableNames', predictorNames);
        naiveBayesPredictFcn = @(x) predict(classificationNaiveBayes, x);
        trainedClassifier.predictFcn = @(x) naiveBayesPredictFcn(predictorExtractionFcn(x));
        
        trainedClassifier.ClassificationNaiveBayes = classificationNaiveBayes;
        partitionedModel = crossval(trainedClassifier.ClassificationNaiveBayes, 'KFold', 10);
        
        [validationPredictions, validationScores] = kfoldPredict(partitionedModel);
        [~,posterior] = kfoldPredict(partitionedModel);
        [fpr,tpr,~,auc] = perfcurve(inputTable.Y, posterior(:,2), partitionedModel.ClassNames(2));
        
        C = confusionmat(response, validationPredictions);
        [val1, val2] = calculateMeasures(C, auc);
        
    otherwise
        disp('Invalid option selected')
        
end

results = {val1, val2, fpr, tpr};

end