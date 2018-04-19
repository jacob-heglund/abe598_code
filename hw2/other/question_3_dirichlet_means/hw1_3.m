clear all;
clc;
close all;

%% load data + initialize parameters
load Gauss_data1.mat
x = Z(1, :);
y = Z(2, :);
data = [x; y];

% number of data points
numData = size(data,2);

% number of initial clusters
k = 1;

% error for stop condition
tolerance = 10^-4;

%% cluster penalty parameter, learn this 
k_approx = 4;

% initialize a set T with the global mean
xAvg = mean(x);
yAvg = mean(y);
T = [xAvg; yAvg];
% for 1:k, find the furthest point from the mean and add it to the set T
dist = zeros(1, numData);

for i = 1:k_approx
    xtAvg = mean(T(1,:)); 
    ytAvg = mean(T(2,:));
    tAvg = [xtAvg; ytAvg];
    % find the furthest point from the mean of the entries of T
    for j = 1:numData
        dist(i) = norm(tAvg - data(:,j), 2);
    end
    
    [val, idx] = max(dist);
    xMax = x(idx);
    yMax = y(idx);
    dataMax = [xMax; yMax];
    T = [T dataMax];
    if i == k_approx
        lambda = val
    end
end
%% 


%% step 1
L1 = data;
% mean1 = [mean(x); mean(y)];

% hack, just set it as a random point in the data
randInteger1 = randi([1, numData]);
randInteger2 = randi([1, numData]);
mean1 = [data(randInteger1); data(randInteger2)];

% each column is the mean x-y values for a different cluster
clusterMeans = [mean1];

%% step 2 - initialize cluster indicators
z = ones(1, numData);

%% step 3 - repeat until convergence
% loop over each data point

d = zeros(k, numData);
done = 0;

while (done == 0)
    for i = 1:numData
        % compute distance between each data point and each cluster

        % each entry of d(c,i) is the distance between a datapoint x_i
        % for i = {1, 2, ..., numData} and cluster with index c = {1, 2, ..., k}
        for c = 1:k
            d(c,i) = norm(data(:,i) - clusterMeans(c), 2)^2;
        end

        for c = 1:k
            [minDist, minDistIdx] = min(d(c,i));
            % if the minimum distance from a datapoint to the mean of a gaussian is
            % larger than some amount lambda, create a new gaussian with the 
            % mean of the coordinates of that datapoint
            if (minDist > lambda)
                k = k+1;
                z(i) = k;
                clusterMeans = [clusterMeans data(:,i)];
            % the cluster indicator for a datapoint becomes the index of the 
            % nearest cluster
            else
                z(i) = minDistIdx;
            end
        end
    end

    % generate clusters
    objarray(k,1) = clusterClass;

    for i = 1:numData
        clusterIdx = z(i);

        for j = 1:k
            objarray(j).members = [objarray(j).members x(:,clusterIdx)];
        end
    end
    
    % find the mean of each cluster
    for i = 1:k
        clusterSize = size(objarray(k).members, 2);
        meanX = mean(objarray(k).members(:,1));
        meanY = mean(objarray(k).members(:,2));
        clusterMeans(k) = [meanX; meanY];
    end
    
    % stop condition
    % Repeat steps 2 and 3 if the centroid positions change betweens
    % steps
    deltaMean = 0;
    
    for i = 1:k
        deltaMean = deltaMean + norm(objarray(i).oldCentroid - objarray(i).centroid, 2);
    end    
    
    if deltaMean < tolerance
        done = 1;
    else
        objarray(i).oldCentroid = objarray(i).centroid;
        objarray(i).members = [];
    end
    
end

















