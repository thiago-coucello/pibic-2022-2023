clc
clear all

files = dir('H:\My Drive\Doutorado\BaseColuna\shared\Datasets\DatasetBalanced2\Results\csvs\DL\refined\*.csv');
csvs{length(files)} = 0;
names{length(files)} = ' ';
mydata(100, length(files)) = 0;
 
for i=1:length(files)
    csvs{i} = readtable(strcat(files(i).folder, '\', files(i).name));
    names{i} = files(i).name;
    s1 = size(csvs{i},1);
    mydata(1:s1, i) = 100*csvs{i}.val_specificity;
%     kstest(csvs{i}.val_accuracy)
end



% return
 
 boxplot(mydata,'Labels',names); hold on
  hold on
 plot(mean(mydata), 'xb')
 hold off
% return
% hold on
% m = mean(Hausdorf)
% plot(m,1:size(Hausdorf,2), '*','r'); 
ax = gca();                 %axis handle (assumes 1 axis)
bph = ax.Children;          %box plot handle (assumes 1 set of boxplots)
bpchil = bph.Children;      %handles to all boxplot elements
bh = findobj(gca,'type','line','tag','Upper Whisker');
UpperWhisker = reshape([bh.YData], [2,length(files)]);

bh = findobj(gca,'type','line','tag','Lower Whisker');
LowerWhisker = reshape([bh.YData], [2,length(files)]);

h = findobj(gcf,'tag','Median');
medians = cell2mat(get(h,'YData'));
medians = [medians(:,1)'];

h = findobj(gcf,'tag','Outliers');
outliers = get(h,'YData');


tam = size(mydata, 1);
size(mydata, 2)
out(size(mydata, 2)) = 0;
trim(size(mydata, 2)) = 0;
m = mean(mydata); % Sample mean
s = std(mydata);
[m; s]


names;
for i=1:size(mydata,2)
    out(i) = size(outliers{i,1},2)/tam;
    trim(i) = trimmean(mydata(:,i), out(i));
end


for i=1:length(files)
    length(files)+1-i;
    
    string = string + '%' + names(length(files)+1-i) + newline + " \addplot [black, fill=red!80, boxplot prepared={lower whisker=" + num2str(LowerWhisker(1,i)) + ...
        ",lower quartile=" + num2str(LowerWhisker(2, i)) + ",upper quartile=" + num2str(UpperWhisker(1,i)) + ...
        ",upper whisker=" + num2str(UpperWhisker(2,i)) + ",median=" + num2str(medians(i)) + ",average=" + num2str(trim(length(files)+1-i)) + ...
        ",sample size=100""}] coordinates {};" + newline;
end
string