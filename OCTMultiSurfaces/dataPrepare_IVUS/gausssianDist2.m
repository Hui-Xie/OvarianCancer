% research Gaussian distribution and its square distribution.
x =[-4:0.1:4];
y1 = normpdf(x,0,1);
y2 = y1.*y1
y2sum= sum(y2)
y2 = y2./y2sum

figure
plot(x,y1,x,y2)
% square of gaussian distribution then normlize is not a gaussian distribution again, 
% with less peak prob and less standard deviation. 