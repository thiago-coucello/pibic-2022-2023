clc
clear all

files1 = dir('H:\My Drive\Doutorado\BaseColuna\shared\Datasets\DatasetBalanced2\Results\csvs\DL\Selecionados\*.csv');
csvs1{length(files1)} = 0;
names1{length(files1)} = ' ';
mydata1(100, length(files1)) = 0;

files2 = dir('H:\My Drive\Doutorado\BaseColuna\shared\Datasets\DatasetBalanced2\Results\csvs\ML\Selecionados\*.csv');
csvs2{length(files2)} = 0;
names2{length(files2)} = ' ';
mydata2(100, length(files2)) = 0;

for i=1:length(files1)
    csvs1{i} = readtable(strcat(files1(i).folder, '\', files1(i).name));
    csvs2{i} = readtable(strcat(files2(i).folder, '\', files2(i).name));
    names1{i} = files1(i).name;
    names2{i} = files2(i).name;
    mydata1(:, i) = csvs1{i}.val_sensitivity;
    mydata2(:, i) = csvs2{i}.val_sensitivity;
    % Is it in a normal distribution?
%     kstest(csvs1{i}.val_sensitivity)
    % Is it in a normal distribution?
%     kstest(csvs2{i}.val_sensitivity)
end

[a b] = max(csvs1{1,3}.val_accuracy)

return

alpha = 0.01;
tail = 'right';

% Not in normal distribution, so test Resnet Against
[p1, h1, stats1] = ranksum(mydata1(:, 3), mydata2(:, 2), 'alpha', alpha, 'tail', tail); % vs Gradient Boosting
[p2, h2, stats2] = ranksum(mydata1(:, 3), mydata2(:, 1), 'alpha', alpha, 'tail', tail); % vs ExtraTrees
[p3, h3, stats3] = ranksum(mydata1(:, 3), mydata2(:, 3), 'alpha', alpha, 'tail', tail); % vs HistoryGradient
[p4, h4, stats4] = ranksum(mydata1(:, 3), mydata2(:, 4), 'alpha', alpha, 'tail', tail); % vs Discriminant Analysis

[p5, h5, stats5] = ranksum(mydata1(:, 3), mydata1(:, 4), 'alpha', alpha, 'tail', tail); % vs XceptionNet
[p6, h6, stats6] = ranksum(mydata1(:, 3), mydata1(:, 2), 'alpha', alpha, 'tail', tail); % vs MobileNet
[p7, h7, stats7] = ranksum(mydata1(:, 3), mydata1(:, 1), 'alpha', alpha, 'tail', tail); % vs DenseNet

[p1; p2; p3; p4; p5; p6; p7]