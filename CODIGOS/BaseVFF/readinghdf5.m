clc
clear all

xception = importKerasLayers('H:\My Drive\Doutorado\BaseColuna\shared\Datasets\DatasetBalanced2\Results\nets\heatmaps\MobileNet_weights72.hdf5', ...
    'ImportWeights', true);

plot(xception)
title('XceptionNet Architecture')