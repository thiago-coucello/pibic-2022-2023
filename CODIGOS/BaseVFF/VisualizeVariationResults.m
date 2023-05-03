% clc
% clear all

load('/Volumes/GoogleDrive/My Drive/Doutorado/BaseColuna/shared/Jonathan/varyingResults.mat');

aucs(6,51) = 0;
for i=1:51
    aucs(1,i) = str2double(AllResults{1,i}{1,1}{1,1}{1,1}(end-3:end));
    aucs(2,i) = str2double(AllResults{1,i}{1,2}{1,1}{1,1}(end-3:end));
    aucs(3,i) = str2double(AllResults{1,i}{1,3}{1,1}{1,1}(end-3:end));
    aucs(4,i) = str2double(AllResults{1,i}{1,4}{1,1}{1,1}(end-3:end));
    aucs(5,i) = str2double(AllResults{1,i}{1,5}{1,1}{1,1}(end-3:end));
    aucs(6,i) = str2double(AllResults{1,i}{1,6}{1,1}{1,1}(end-3:end));
end

[a b] = max(aucs(1,:))

plot(aucs(1,:)); hold on
[i val] = max(aucs(1,:))
plot(aucs(2,:)); hold on
plot(aucs(3,:)); hold on
plot(aucs(4,:)); hold on
plot(aucs(5,:)); hold on
plot(aucs(6,:)); hold on