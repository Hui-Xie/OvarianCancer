%{
code is borrowed from Leixin Zhou batch_metric.m, courtesy to Leixn Zhou.
compute luemen and media performance of IVUS data set
%}

clear all
clc

predDir = '/home/hxie1/data/IVUS/polarNumpy/log/SurfacesUnet/expUnetIVUS_Lace_Scale_TTA_20200219/testResult/text';
gtDir = '/home/hxie1/data/IVUS/Test_Set/Data_set_B/LABELS_obs1';
%gtDir = '/home/hxie1/data/IVUS/Test_Set/Data_set_B/LABELS_obs2_v1'; %Leixin used. It is best.
%gtDir = '/home/hxie1/data/IVUS/Test_Set/Data_set_B/LABELS_obs2_v2';
predictLumenFiles = dir(fullfile(predDir,'lum_*_003.txt'));
predictLumenFiles = {predictLumenFiles.name}';
N = numel(predictLumenFiles);
fprintf("Test set has %d images comparing with ground truth at %s.\n", N, gtDir)

% result data tables
lumenDataTable = zeros(N, 4);
mediaDataTable = zeros(N, 4);
% column: jacc, dice, hd, pad
% row: different file

for i=1:N
    predLumenFile = fullfile(predDir, predictLumenFiles{i});
    gtLumenFile = fullfile(gtDir, predictLumenFiles{i});
    prediction = dlmread(predLumenFile, ',')';
    gt = dlmread(gtLumenFile, ',')';
    [lumenDataTable(i,1), lumenDataTable(i,2), lumenDataTable(i,3), lumenDataTable(i,4)]= computePerformance_cart(prediction, gt, 384);
    
    predMediaFile = strrep(predLumenFile,'lum_','med_');
    gtMediaFile = strrep(gtLumenFile,'lum_','med_');
    prediction = dlmread(predMediaFile, ',')';
    gt = dlmread(gtMediaFile, ',')';
    [mediaDataTable(i,1), mediaDataTable(i,2), mediaDataTable(i,3), mediaDataTable(i,4)]= computePerformance_cart(prediction, gt, 384);   
    
end

% output result
lumenMean = mean(lumenDataTable,1);
lumenStd = std(lumenDataTable,1);
mediaMean = mean(mediaDataTable,1);
mediaStd = std(mediaDataTable,1);
fprintf("\tJacc\t\tDice\t\tHD\t\tPAD\n")
fprintf("Lumen:\t")
for i=1:4
    fprintf("%.2f%c%.2f\t", lumenMean(i), char(177),lumenStd(i))    
end
fprintf("\nMedia:\t")
for i=1:4
    fprintf("%.2f%c%.2f\t", mediaMean(i), char(177),mediaStd(i))    
end
fprintf("\n\n")

