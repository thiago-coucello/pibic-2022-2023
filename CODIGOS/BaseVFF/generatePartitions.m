clc
clear all

DatasetName = 'DatasetBalanced';

D = dir(strcat('E:\Datasets\', DatasetName ,'\*\*.bmp'));
D = struct2table(D);
D = sortrows(D,'name');
size(D)
load('H:\My Drive\Doutorado\BaseColuna\fraturados.mat')
load('H:\My Drive\Doutorado\BaseColuna\classes-info.mat')

% noFracture = a == 2 | a == 4;
% Fracture = a == 3 | a == 5;

Class(size(D,1)) = 0; 
Exam(size(D,1)) = 0; 
Sizes(400,3) = 0;

cont = 1;
for tw=1:2
    for i=1:size(a,1)
        for j=1:5
            if (fraturados(i,j) == 0 && a(i,1) > 1)
    %             [i, j cont]
                classe = (a(i,1) == 3 || a(i,1) == 5);
                Sizes(cont, 2) = classe;
                Sizes(cont, 3) = str2num(strcat(num2str(tw), num2str(i), '00', num2str(j)));
                cont = cont + 1;
             end
        end
    end
end



cvp{100} = 0;


for k=1:100
    k
    clear Train Test Image Class tb filename
    Image{size(D,1),1} = ' ';
    Class(size(D,1),1) = false;
    Train(size(D,1),1) = false;
    Test(size(D,1),1) = false;
    Ids(size(D,1),1) = 0;
    
    if (k == 1)
        cvp{k} = cvpartition(Sizes(:, 2),'Holdout', 0.2);
    else
        cont = 0;
        while 1
            cvp{k} = repartition(cvp{k-1});
            for kk=1:k-1
                stats = isequal(cvp{kk}.test, cvp{k}.test);
            end
            if (~stats)
                break;
            end
            cont = cont + 1;
        end
    end
    
    
    cvpTrain = cvp{k}.training;
    cvpTest = cvp{k}.test;

    for i=1:size(D,1)
        mystr = char(D.folder(i));
        myexam1 = char(D.name(i));
        myVertebra = str2num(myexam1(10:11));
        myexam = str2num(myexam1(6:7));
        tw = str2num(myexam1(2:3));
        Image{i} = strcat('../Datasets/', DatasetName, '/', mystr(end-6:end), '/', D.name(i));
        Class(i,1) = logical(str2num(mystr(end:end)));
        
        
       
        Ids(i) = str2num(strcat(num2str(tw), num2str(myexam), '00', num2str(myVertebra)));
        
        idx = find(Ids(i) == Sizes(:, 3) );
        if (Class(i,1) ~= Sizes(idx, 2) || length(Ids(i)) > 1)
            disp('something went wrong')
        end
        
        if (cvpTrain(idx) == 1)
            Train(i) = 1;
        elseif (cvpTest(idx) == 1)
            Test(i) = 1;
        end
    end
    
%     sum(Test & Class)
%     sum(Test & ~Class)
%     
%     sum(Train & Class)
%     sum(Train & ~Class)
    
    tb = table(Image,Class,Train,Test);
%   return


    filename = strcat('E:\Datasets\', DatasetName, '\Partitions\',num2str(k,'%2d'), '.csv');
    writetable(tb,filename)
end

