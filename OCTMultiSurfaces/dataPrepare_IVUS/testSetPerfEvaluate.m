%{
code is borrowed from Leixin Zhou batch_metric.m, courtesy to Leixn Zhou.
compute luemen and media performance of IVUS data set

If you need to execute a matlab script you can do matlab -nodisplay < script.m.
If you want to call a matlab function, you can do matlab -nodisplay -r "foo(); quit".
IIRC, in both cases matlab should exit once it is done.




%}
clc
predDirList=["/raid001/users/hxie1/data/IVUS/polarNumpy/log/SurfaceSubnet_Q_BEMA/expIVUS_20210614_SurfaceSubnetQ32_100percent_NOBEMA_skm2/testResult/text", ...
"/raid001/users/hxie1/data/IVUS/polarNumpy/log/SurfaceSubnet_Q_BEMA/expIVUS_20210614_SurfaceSubnetQ32_100percent_NOYweight_skm2/testResult/text", ...
"/raid001/users/hxie1/data/IVUS/polarNumpy/log/SurfaceSubnet_Q_BEMA/expIVUS_20210614_SurfaceSubnetQ32_100percent_NOTopo_skm2/testResult/text", ...
"/raid001/users/hxie1/data/IVUS/polarNumpy/log/SurfaceSubnet_Q_BEMA/expIVUS_20210614_SurfaceSubnetQ32_100percent_NOGradientInput_skm2/testResult/text"];
NPred = length(predDirList);

for i= 1:NPred
    predDir = predDirList(i);
    evaluePred(predDir);
end


function result=evaluePred(predDir)
    result = 0;

    % predDir = '/raid001/users/hxie1/data/IVUS/polarNumpy/log/SurfaceSubnet_Q/expIVUS_20210514_SurfaceSubnetQ64_100percent_A_skm2/testResult/text';
    % predDir = '/raid001/users/hxie1/data/IVUS/polarNumpy/log/SurfacesUnet_YufanHe_2/expIVUS_20210514_YufanHe_100percent_A_skm2/testResult/text'
    % predDir = '/raid001/users/hxie1/data/IVUS/polarNumpy/log/SurfacesUnet/expUnetIVUS_Sigma0_20200302/realtime_testResult/text';
    % predDir = "/raid001/users/hxie1/data/IVUS/polarNumpy_10percent/log/SurfaceSubnet_Q/expIVUS_20210514_SurfaceSubnetQ64_10percent_A_skm2/testResult/text"
    % predDir = "/raid001/users/hxie1/data/IVUS/polarNumpy_10percent/log/SurfacesUnet_YufanHe_2/expIVUS_20210514_YufanHe_10percent_A_skm2/testResult/text"

    %gtDir = '/raid001/users/hxie1/data/IVUS/Test_Set/Data_set_B/LABELS_obs1';
    gtDir = '/raid001/users/hxie1/data/IVUS/Test_Set/Data_set_B/LABELS_obs2_v1'; %Leixin used. It is best.
    %gtDir = '/raid001/users/hxie1/data/IVUS/Test_Set/Data_set_B/LABELS_obs2_v2';
    predictLumenFiles = dir(fullfile(predDir,'lum_*_003.txt'));
    predictLumenFiles = {predictLumenFiles.name}';
    N = numel(predictLumenFiles);
    fprintf("===============================================================================================================\n")
    fprintf("PredictionDir: %s\n", predDir)
    fprintf("GroundTruthDir: %s\n", gtDir)
    fprintf("Test set size: %d images \n", N)

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
    fprintf("\t\tJacc\t\tDice\t\tHD\t\tPAD\n")
    fprintf("Lumen:\t")
    for i=1:4
        fprintf("%.2f%c%.2f\t", lumenMean(i), char(177),lumenStd(i))
    end
    fprintf("\nMedia:\t")
    for i=1:4
        fprintf("%.2f%c%.2f\t", mediaMean(i), char(177),mediaStd(i))
    end
    fprintf("\n\n")
    fprintf("================================================================================================================\n\n")

end




