%  analyze Duke OCT file
clear all
clc

amdFile = "/home/sheen/temp/Duke/Farsiu_Ophthalmology_2013_AMD_Subject_1174.mat";
controlFile = "/home/sheen/temp/Duke/Farsiu_Ophthalmology_2013_Control_Subject_1056.mat";

patient = load(controlFile)
% patient:
%        images: [512×1000×100 double]: Height*Width*Slice
%     layerMaps: [100×1000×3 double]  : 
%             Slice*Widht* NumSurface, 
%             extract middle volume: 60 slices, Width 400, height 512.
%             width index: 299-698, slice index: 21-80, height index:1-512
%                   
%           Age: 77

% define non-NaN range of layerMaps
for s=1:100
    layer = squeeze(patient.layerMaps(s,:,:));
    NonNan = not(isnan(layer));
    if sum(NonNan) ==0
        continue;
    end
    for surface=1:3
        line=NonNan(:,surface);
        if sum(line)==0
            continue;
        end    
        nonNanIndex = find(line);
        fprintf("s=%d, surface=%d, min=%d, max=%d\n",s,surface,nonNanIndex(1),nonNanIndex(end))        
    end
    
end

s = 50; % slice
imshow(patient.images(:,:,s),[])
hold on
X = 299:1:698; % the middle width of 512 of width 1000 image
lineColor=['r','g','b'];
for i=1:3
    Y = patient.layerMaps(s,X,i);
    plot(X,Y,lineColor(i));
end

disp("================")
