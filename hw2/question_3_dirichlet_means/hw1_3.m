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

% learn the cluster penalty parameter lambda
k_approx = 4;
lambda = findLambda(x,y,k_approx);

%% Dirichlet Process Clustering
% number of initial clusters
k = 1;

% error for stop condition
tolerance = 10^-4;

% step 1
L1 = data;
mean1 = [mean(x); mean(y)];

% hack, just set it as a random point in the data
%randInteger1 = randi([1, numData]);
%randInteger2 = randi([1, numData]);
%mean1 = [data(randInteger1); data(randInteger2)];

% each column is the mean x-y values for a different cluster
clusterMeans = [mean1];

%% step 2 - initialize cluster indicators
z = ones(1, numData);

%% step 3 - repeat until convergence
% loop over each data point

done = 0;
while (done == 0)
    for i = 1:numData
        [z,k] = dataLabel(data, clusterMeans, lambda, k);
    end
    
    %{
    % generate clusters
    objarray(k,1) = clusterClass;

    for i = 1:numData
        clusterIdx = z(i);

        for j = 1:k
            objarray(j).members = [objarray(j).members x(:,clusterIdx)];
        end
    end
    %}
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
















