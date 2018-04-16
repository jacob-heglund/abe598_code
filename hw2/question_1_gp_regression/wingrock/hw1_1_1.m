clear all;
clc;

% load the data
% the order of the colums is [PDOT_REC  X_REC  XDOT_REC); 
data1 = csvread('sim1.csv');
data2 = csvread('sim2.csv');
data3 = csvread('sim3.csv');
data4 = csvread('sim4.csv');
data5 = csvread('sim5.csv');

% concatenate all sims into one matrix
data = [data1; data2; data3; data4; data5];

% [1; x(1); x(2); abs(x(1))*x(2); abs(x(2))*x(2); x(1)^3];
% x(1) = phi = X_REC = data(:,2)
% x(2) = p = XDOT_REC = data(:,3)

pDot = data(:,1);
phi = data(:,2);
p = data(:,3);

% do multivariate regression
% input to calculate pDot
numData = size(data,1);
first = ones(numData,1);

% give the basis augmented by a column of ones as the first column
X = [first, phi, p, abs(phi).*p, abs(p).*p, phi.^3];

% output: pDot
Y = data(:,1);

% find basis coefficients
[b] = regress(Y,X);
















