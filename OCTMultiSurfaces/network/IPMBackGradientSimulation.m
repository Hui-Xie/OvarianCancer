% simulation of IPM backward gradient
% matlab index start from 1.
% optimization problem:
% s* = argmin_s { 0.5*s'*H*s + b'*s }, such that A*s<=d
clear all;
clc;
disp('IPM backward gradient simulation experiment')

%for notGoHere=[]  %case 1:
%quadratic equation parameters
disp("For perfect s* case:")
H = [1,2;3,5];
b = [-1,-2.6]';
A = -[1,0; 0,1];
d = [0,0]';

%s, lambda,and r all got from IPM optimization forward
s = [0.2,0.4]'; % s*, 
lambda = [1.3316e-04, 2.9466e-05]';
r =[-2.8254e-05, 1.6072e-05, 1.4850e-05, 6.9906e-08]';  %resiudal vector
t = 861875;
%end 

for notGohere=[] % case 2:
% quadratic equation parameters
disp("For un-symmetric H case")
H = [2,5;3,11];
b = [3,5]';
A = [5,2; 3,4];
d = [2,3]';

% s, lambda,and r all got from IPM optimization forward
s = [-1.1428,-0.1429]'; % s*, 
lambda = [2.6280e-08,1.5542e-07]';
r =[7.4031e-05, 5.9802e-05, -1.0317e-06, -1.5407e-07]';  %resiudal vector
t = 861875.0625;
end

for notGoHere=[] % case 3:
% quadratic equation parameters
disp("For symmetric H case")
H = [2,5;5,11];
b = [3,5]';
A = [5,2; 3,4];
d = [2,3]';

% s, lambda,and r all got from IPM optimization forward
s = [0.6744,-0.6859]'; % s*, 
lambda = [3.9717e-07,1.9743e-03]';
r =[9.2500e-01, 8.3449e-01, -3.3422e-04, 7.0113e-03]';  %resiudal vector
t = 2992.0208;
end

% =============================
dL_ds = [3,15]';
assert(all([2,2] == size(H)))

%===========Manual Computation of dL/dH===============
%== by residula equataion system==========:
syms s1(h1,h2,h3,h4,b1,b2, a1,a2,a3,a4, d1,d2)
syms s2(h1,h2,h3,h4,b1,b2, a1,a2,a3,a4, d1,d2)
s1(h1,h2,h3,h4,b1,b2, a1,a2,a3,a4, d1,d2) = (1/((h1-lambda(1)*a1)*(h4-lambda(2)*a4)-(h2-lambda(1)*a2)*(h3-lambda(2)*a3)))...
*((h4-lambda(2)*a4)*(r(1)+r(3)+1/t-b1-lambda(1)*d1-a1*lambda(1)-a3*lambda(2))+(-h2+lambda(1)*a2)*(r(2)+r(4)+1/t-b2-lambda(2)*d2-a2*lambda(1)-a4*lambda(2)));

s2(h1,h2,h3,h4,b1,b2, a1,a2,a3,a4, d1,d2) = (1/((h1-lambda(1)*a1)*(h4-lambda(2)*a4)-(h2-lambda(1)*a2)*(h3-lambda(2)*a3)))...
*((-h3+lambda(2)*a3)*(r(1)+r(3)+1/t-b1-lambda(1)*d1-a1*lambda(1)-a3*lambda(2))+(h1-lambda(1)*a1)*(r(2)+r(4)+1/t-b2-lambda(2)*d2-a2*lambda(1)-a4*lambda(2)));


for notGoHere = []
    % stationary condition
    %   s= H^(-1)*(r-b-A'*lambda)
    %   H = [h1,h2; h3,h4]
    %   A = [a1,a2; a3,a4]
    
    syms s1(h1,h2,h3,h4,b1,b2, a1,a2,a3,a4);
    s1(h1,h2,h3,h4,b1,b2, a1,a2,a3,a4) = (h4*(r(1)-b1-a1*lambda(1)-a3*lambda(2))-h2*(r(2)-b2-a2*lambda(1)-a4*lambda(2)))/(h1*h4-h2*h3);
    
    syms s2(h1,h2,h3,h4,b1,b2, a1,a2,a3,a4);
    s2(h1,h2,h3,h4,b1,b2, a1,a2,a3,a4) = (-h3*(r(1)-b1-a1*lambda(1)-a3*lambda(2))+h1*(r(2)-b2-a2*lambda(1)-a4*lambda(2)))/(h1*h4-h2*h3);
end

ds1_dh1 = diff(s1,h1);
ds1_dh2 = diff(s1,h2);
ds1_dh3 = diff(s1,h3);
ds1_dh4 = diff(s1,h4);
ds1_dH = [ds1_dh1, ds1_dh2; ds1_dh3, ds1_dh4];

ds2_dh1 = diff(s2,h1);
ds2_dh2 = diff(s2,h2);
ds2_dh3 = diff(s2,h3);
ds2_dh4 = diff(s2,h4);
ds2_dH = [ds2_dh1, ds2_dh2; ds2_dh3, ds2_dh4];

dL_dH = dL_ds(1)* ds1_dH + dL_ds(2)*ds2_dH;

ds1_da1 = diff(s1,a1);
ds1_da2 = diff(s1,a2);
ds1_da3 = diff(s1,a3);
ds1_da4 = diff(s1,a4);
ds1_dA = [ds1_da1, ds1_da2; ds1_da3, ds1_da4];

ds2_da1 = diff(s2,a1);
ds2_da2 = diff(s2,a2);
ds2_da3 = diff(s2,a3);
ds2_da4 = diff(s2,a4);
ds2_dA = [ds2_da1, ds2_da2; ds2_da3, ds2_da4];

dL_dA = dL_ds(1)* ds1_dA + dL_ds(2)*ds2_dA;  

%disp("dL_dH in manual computation:")
manual_dL_dH = double(dL_dH(H(1,1),H(1,2),H(2,1),H(2,2),b(1),b(2), A(1,1),A(1,2),A(2,1),A(2,2),d(1),d(2)));
manual_dL_dA = double(dL_dA(H(1,1),H(1,2),H(2,1),H(2,2),b(1),b(2), A(1,1),A(1,2),A(2,1),A(2,2),d(1),d(2)));


%===========Compute J, ds,dlambda================================
% compute Jocabian
J = [H, A'; -diag(lambda)*A, -diag(A*s-d)];
dslambda = -(inv(J))'*[dL_ds;0;0];
ds = dslambda(1:2);
dlambda = dslambda(3:4);

%===========Computation of dL/dH using Hui's formula===============
%disp("dL_dH in Hui's formula:")
hui_dL_dH = diag(ds)*[s';s'];
hui_dL_dA = diag(lambda)*([ds';ds']-diag(dlambda)*[s';s']);

%===========Computation of dL/dH using Amos' fomula===============
%disp("dL_dH in Amos's formula:")
amos_dL_dH = 0.5*(ds*s'+ s*ds');
amos_dL_dA = diag(lambda)*(dlambda*s'+ lambda*ds');

%===========display=========================
disp('optimization problem parameters:')
H
b
A
d
s
lambda
r
t
dL_ds

disp('====================================')
disp('dL_dH: the backward gradient on quadratic matrix:')
manual_dL_dH
hui_dL_dH
amos_dL_dH
disp('====================================')
disp('dL_dA: the backward gradient on constraint matrix:')
manual_dL_dA
hui_dL_dA
amos_dL_dA
disp('====================================')







