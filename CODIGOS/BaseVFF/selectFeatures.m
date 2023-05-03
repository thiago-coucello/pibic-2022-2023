

function [data, escolhidos, scores] = selectFeatures(data, thresh)

    classe = data(:, end);
    data2 = data;
    data2(:, end) = [];
    [idx1,scores] = fscchi2(data2, classe);
    scores1 = scores;
    

    mymin = min(scores1);
    mymax = max(scores1);

    if (mymin < 0)
        scores1 = scores1 + ( -1*mymin );
    end

    scores1 = scores1/mymax;
    scores = scores1;

    escolhidos = scores >= thresh;
%     find(scores1 >= thresh)
    data = data(:, escolhidos'==1);

    data = cat(2, data, classe);


end