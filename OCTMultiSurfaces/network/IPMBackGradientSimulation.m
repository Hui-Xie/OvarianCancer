% simulation of IPM backward gradient
% matlab index start from 1.
% optimization problem:
% s* = argmin_s { 0.5*s'*H*s + b'*s }, such that A*s<=d
clear all;
clc;
disp('IPM backward gradient simulation experiment')

H = [1,2;3,5];
b = [-1,-2.6]';
s = [0.2,0.4]'; % s*, gotten by optimization forward
A = -[1,0; 0,1];
d = [0,0]';
lambda = [0.002, 0.001]';
t = 2500;
epsilon =[-0.002, -0.001,0,0]';

dL_ds = [3,7]';
assert(all([2,2] == size(H)))

%===========Manual Computation of dL/dH===============
%== by residula equataion:
%   s= H^(-1)*(epsilon-b+lambda)
%start s symbolic function with H = [h1,h2; h3,h4]
syms s1(h1,h2,h3,h4,b1,b2);
s1(h1,h2,h3,h4,b1,b2) = -(h4*b1-h2*b2)/(h1*h4-h2*h3);
ds1_dh1 = diff(s1,h1);
ds1_dh2 = diff(s1,h2);
ds1_dh3 = diff(s1,h3);
ds1_dh4 = diff(s1,h4);
ds1_dH = [ds1_dh1, ds1_dh2; ds1_dh3, ds1_dh4];

syms s2(h1,h2,h3,h4,b1,b2);
s2(h1,h2,h3,h4,b1,b2) = -(-h3*b1+h1*b2)/(h1*h4-h2*h3);
ds2_dh1 = diff(s2,h1);
ds2_dh2 = diff(s2,h2);
ds2_dh3 = diff(s2,h3);
ds2_dh4 = diff(s2,h4);
ds2_dH = [ds2_dh1, ds2_dh2; ds2_dh3, ds2_dh4];

dL_dH = dL_ds(1)* ds1_dH + dL_ds(2)*ds2_dH;  

%disp("dL_dH in manual computation:")
manual_dL_dH = dL_dH(H(1,1),H(1,2),H(2,1),H(2,2),b(1),b(2))

%===========Compute J, ds,dlambda================================
% compute Jocabian
J = [H, A'; -diag(lambda)*A, -diag(A*s-d)];
dslambda = -(inv(J))'*[dL_ds;0;0];
ds = dslambda(1:2);
dlambda = dslambda(3:4);

%===========Computation of dL/dH using Hui's formula===============
%disp("dL_dH in Hui's formula:")
hui_dL_dH = diag(ds)*[s';s']

%===========Computation of dL/dH using Amos' fomula===============
%disp("dL_dH in Amos's formula:")
amos_dL_dH = 0.5*(ds*s'+ s*ds')









