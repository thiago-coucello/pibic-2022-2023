
myPath = 'G:\Meu Drive\Doutorado\BaseColuna\shared\Datasets\DatasetBalanced2\Features\features.csv';
T = readtable(myPath);

T(:, 1:3) = [];
T(:,2) = [];
classe = T(:,1);
T(:,1) = [];

namesAll = T.Properties.VariableNames;


[idx1,scores] = fscchi2(T, classe);

scores = scores/max(scores);

h = hist(scores, 10);

Name{216,1} = ' ';
ChiSquareRanking(216,1) = 0;

contador = 1;

for i=1:length(idx1)
    if (scores(idx1(i)) >= 0.2)
        Name{contador} = namesAll{idx1(i)};
        ChiSquareRanking(contador) = scores(idx1(i));
        contador = contador + 1;
    end
end

selectedFeatures = table(Name,ChiSquareRanking);

writetable(selectedFeatures, 'SelectedFeatures02.csv')