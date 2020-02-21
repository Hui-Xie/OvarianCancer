% research Gaussian distribution and its square distribution.
x =[-4:0.1:4];
y1 = normpdf(x,0,1);
y1sum = sum(y1);
y1 = y1./y1sum;

y2 = y1.*y1;
y2sum= sum(y2);
y2 = y2./y2sum;

figure
plot(x,y1,"r-",x,y2,"g*")
% conclusion:
% square of gaussian distribution then normlize is still a gaussian distribution again, 
% with bigger peak prob and less standard deviation. 