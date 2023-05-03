

function [val1, val2] = calculateMeasures(C, ROC)


TP = C(1,1);
FP = C(2,1);
TN = C(2,2);
FN = C(1,2);
FPR = FP/(FP+TN);
FNR = FN/(FN+TP);
        
Acc = (TP+TN)/(TP+TN+FP+FN);
Pre =  TP/(TP+FP);
Rec = TP/(TP+FN);
Spec = TN/(TN+FP);
Fm = (2*Pre*Rec)/(Pre+Rec);
AUC = ROC;
% 1 - ((FPR+FNR)/2);


val1 = strcat(num2str(Acc*100,'%2.1f'), {' & '}, num2str(Pre*100,'%2.1f'), {' & '}, ...
    num2str(Rec*100,'%2.1f'), {' & '}, num2str(Spec*100,'%2.1f'), {' & '}, ...
num2str(Fm*100,'%2.1f'), {' & '}, num2str(100*ROC,'%2.1f'));

val2 = [Acc, Pre, Rec, Fm, AUC, Spec];