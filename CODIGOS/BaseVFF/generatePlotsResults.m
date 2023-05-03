clc
clear all

load('G:\My Drive\Doutorado\BaseColuna\shared\Jonathan\Results\TestTrainResults22.mat');

accs(100,6) = 0;
prec(100,6) = 0;
recs(100,6) = 0;
fms(100,6) = 0;
aucs(100,6) = 0;
specs(100,6) = 0;

% 1 - Acuracy
% 2 - Precision
% 3 - Recall
% 4 - FMeasure
% 5 - AUC
% 6 - Specificity

for i=1:100
     measure = 1;
    accs(i, 1:6) = [AllResults{1,i}{1,1}{1,2}(measure), AllResults{1,i}{1,2}{1,2}(measure), ...
        AllResults{1,i}{1,3}{1,2}(measure), AllResults{1,i}{1,4}{1,2}(measure), ...
        AllResults{1,i}{1,5}{1,2}(measure), AllResults{1,i}{1,6}{1,2}(measure)];
    
     measure = 2;
    prec(i, 1:6) = [AllResults{1,i}{1,1}{1,2}(measure), AllResults{1,i}{1,2}{1,2}(measure), ...
        AllResults{1,i}{1,3}{1,2}(measure), AllResults{1,i}{1,4}{1,2}(measure), ...
        AllResults{1,i}{1,5}{1,2}(measure), AllResults{1,i}{1,6}{1,2}(measure)];
    
     measure = 3;
    recs(i, 1:6) = [AllResults{1,i}{1,1}{1,2}(measure), AllResults{1,i}{1,2}{1,2}(measure), ...
        AllResults{1,i}{1,3}{1,2}(measure), AllResults{1,i}{1,4}{1,2}(measure), ...
        AllResults{1,i}{1,5}{1,2}(measure), AllResults{1,i}{1,6}{1,2}(measure)];
    
     measure = 4;
    fms(i, 1:6) = [AllResults{1,i}{1,1}{1,2}(measure), AllResults{1,i}{1,2}{1,2}(measure), ...
        AllResults{1,i}{1,3}{1,2}(measure), AllResults{1,i}{1,4}{1,2}(measure), ...
        AllResults{1,i}{1,5}{1,2}(measure), AllResults{1,i}{1,6}{1,2}(measure)];
    measure = 5;
    aucs(i, 1:6) = [AllResults{1,i}{1,1}{1,2}(measure), AllResults{1,i}{1,2}{1,2}(measure), ...
        AllResults{1,i}{1,3}{1,2}(measure), AllResults{1,i}{1,4}{1,2}(measure), ...
        AllResults{1,i}{1,5}{1,2}(measure), AllResults{1,i}{1,6}{1,2}(measure)];
    
    measure = 6;
    specs(i, 1:6) = [AllResults{1,i}{1,1}{1,2}(measure), AllResults{1,i}{1,2}{1,2}(measure), ...
        AllResults{1,i}{1,3}{1,2}(measure), AllResults{1,i}{1,4}{1,2}(measure), ...
        AllResults{1,i}{1,5}{1,2}(measure), AllResults{1,i}{1,6}{1,2}(measure)];
end


[mean(accs); mean(prec); mean(recs); mean(fms); mean(aucs)] 

mydata = fms;
% medida = 10;
% increment = 10;
% % Hausdorf = [FCS(:, medida); CS(:, medida); BG(:, medida); FGC(:, medida); GC(:, medida);FCS(:, medida+10); CS(:, medida+10); BG(:, medida+10); FGC(:, medida+10); GC(:, medida+10)];
% Hausdorf = [    FCS(:, medida+increment),CS(:, medida+increment),BG(:, medida+increment),FGC(:, medida+increment),GC(:, medida+increment),...
%     FCS(:, medida),CS(:, medida), BG(:, medida), FGC(:, medida) , GC(:, medida)];

boxplot(mydata,'Labels',{'Ensemble','KNN','SVM','Tree','Discriminant', 'NaiveBayes'});
% return
% hold on
% m = mean(Hausdorf)
% plot(m,1:size(Hausdorf,2), '*','r'); 
ax = gca();                 %axis handle (assumes 1 axis)
bph = ax.Children;          %box plot handle (assumes 1 set of boxplots)
bpchil = bph.Children;      %handles to all boxplot elements
bh = findobj(gca,'type','line','tag','Upper Whisker');
UpperWhisker = reshape([bh.YData], [2,6]);

bh = findobj(gca,'type','line','tag','Lower Whisker');
LowerWhisker = reshape([bh.YData], [2,6]);

h = findobj(gcf,'tag','Median');
medians = cell2mat(get(h,'YData'))
medians = [medians(:,1)'];

h = findobj(gcf,'tag','Outliers');
outliers = get(h,'YData');

tam = size(mydata,1);
outliers = ([size(outliers{1,1},2), size(outliers{2,1},2), size(outliers{3,1},2), size(outliers{4,1},2), size(outliers{5,1},2), size(outliers{6,1},2)]/tam);

m = mean(mydata); % Sample mean
trim = [trimmean(mydata(:,1), outliers(1)), trimmean(mydata(:,2), outliers(2)), ...
    trimmean(mydata(:,3), outliers(3)), trimmean(mydata(:,4), outliers(4)), ... 
   trimmean(mydata(:,5), outliers(5)), trimmean(mydata(:,6), outliers(6))]; % Trimmed mean

string = "";
fill = {'red!80',...
    'red!80',...
        'red!80',...
            'red!80',...
'red!80',...
'red!80'};

box = {  ',every box/.style={postaction={pattern = north west lines, pattern color=white}}',...
    ',every box/.style={postaction={pattern = crosshatch dots, pattern color=white}}',...
        ',every box/.style={postaction={pattern = vertical lines, pattern color=white}}',...
          ',every box/.style={postaction={pattern = crosshatch, pattern color=white}}',...
',every box/.style={postaction={pattern = sixpointed stars, pattern color=white}}',...
  ',every box/.style={postaction={pattern = north west lines, pattern color=white}}'};
for i=1:6
    string = strcat(string, " \addplot+ [black, fill=", fill(i),", boxplot prepared={lower whisker=", num2str(LowerWhisker(1,i)), ...
        ",lower quartile=",num2str(LowerWhisker(2, i)),",upper quartile=",num2str(UpperWhisker(1,i)),...
        ",upper whisker=",num2str(UpperWhisker(2,i)),",median=",num2str(medians(i)),",average=",num2str(trim(7-i)), ...
        ",sample size=100","}] coordinates {};");
end
