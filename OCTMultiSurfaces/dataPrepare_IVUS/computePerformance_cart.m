function [jacc, dice, hd, pad] = computePerformance(AutoCart,GtCart,size_image)
% courtesy to http://www.cvc.uab.es/IVUSchallenge2011/index-2.html
% created: 27/07/2011
%  Simone Balocco @ Universitat de Barcelona

%  PARAMETERS:
%  size_image is 512 in Boston Scientific Data-set;
%  size_image is 384 in Volcano Data-set;
            
%  PIXEL SIZE:
%  pixel size is 0.0352 mm in Boston Scientific Data-set;
%  size_image is 0.026 mm in Volcano Data-set;


if(nargin<3)
    size_image = 566; % 256 x 256 polar > 566 x 566 cartesian (27 catheter, 256 radius)
end

visual=0;

% usage: RESULTS = computePerformance(yAuto,yObs1,yObs2,dCat)
areaAuto = zeros(size_image,size_image);
areaGT = zeros(size_image,size_image);
AutoCart(:,end+1)=AutoCart(:,1);  %THIS IS IF THE CIRCLE IS NOT PERFECTLY CLOSED
GtCart(:,end+1)=GtCart(:,1);      %THIS IS IF THE CIRCLE IS NOT PERFECTLY CLOSED

GtCartS=spline(1:length(GtCart),GtCart,1:.02:length(GtCart));
for i = 1:length(GtCartS)
    areaGT(round(GtCartS(1,i)),round(GtCartS(2,i))) = 1;    
end
AutoCartS=spline(1:length(AutoCart),AutoCart,1:length(AutoCart)/length(GtCartS):length(AutoCart));
AutoCartS = max(1,min(383, AutoCartS));
for i = 1:length(AutoCartS)
    areaAuto(round(AutoCartS(1,i)),round(AutoCartS(2,i))) = 1;    
end
areaAuto=imfill(areaAuto,'holes');
areaGT=imfill(areaGT,'holes');
if (areaAuto(size_image/2,size_image/2)~=1) && (areaGT(size_image/2,size_image/2)~=1)
    disp('ERROR: insufficient number of interpolating point');
    keyboard;
    return
end
if visual==1
    imagesc(areaGT+areaAuto)
    pause
end
jacc = sum(sum(areaAuto & areaGT))/(sum(sum(areaAuto | areaGT)));

dice = 2.*sum(sum(areaAuto & areaGT))/(sum(sum(areaAuto)) + sum(sum(areaGT)));

% hausdorff distance
hd = 0.026 * HausdorffDist([GtCart(1,:)' GtCart(2,:)'],[AutoCart(1,:)' AutoCart(2,:)']); %CARTESIAN

% area error
%RESULTS.ae = abs(sum(sum(areaAuto)) - sum(sum(areaGT)));

% PAD 
pad = abs(sum(sum(areaAuto)) - sum(sum(areaGT)))/sum(sum(areaGT));




