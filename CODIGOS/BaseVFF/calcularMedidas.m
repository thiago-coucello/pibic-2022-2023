
clc
clear all
     C = [200 118; 166 248];
     
     
     
        TP = C(1,1);
        FP = C(2,1);
        TN = C(2,2);
        FN = C(1,2);
        FPR = FP/(FP+TN);
        FNR = FN/(FN+TP);
        
        ACC = (TP+TN)/(TP+TN+FP+FN);
        PRE = TP/(TP+FP);
        SEN = TP/(TP+FN);
        SPE = TN/(TN+FP);
        FM = (2*PRE*SEN)/(PRE+SEN);
        AUC = 0.871;
        
        
        round(100*[ACC PRE SEN SPE FM AUC], 0)