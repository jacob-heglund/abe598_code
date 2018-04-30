% function returns the cluster labels for each datapoint in data
%inputs
% data - a 2d array of data with size 2 x numDataPts
% clusterMeans - a 2D array of the means of the existing clusters with
% each column the mean for the corresponding cluster index
% i.e. clusterMeans(:,2) is the mean of the 2nd cluster we create

% outputs
% z - the cluster label of each datapoint

function [z,k] = dataLabel(data, clusterMeans, lambda,k)
numDataPts = size(data,2);
numClusters = size(clusterMeans,2);
d = zeros(1,numClusters);

% array of cluster indicators
z = zeros(1,numDataPts);

for i = 1:numDataPts
    dataPt = data(:,i);
    % find the distance squared (so its always positive)
    % from point x_i to the center (mean value) of each cluster
    for c = 1:numClusters
        d(c) = norm(dataPt - clusterMeans(:,c), 2)^2;
    end
    
    [minDist, minDistIdx] = min(d);
    if (minDist > lambda)
        k = k+1;
        z(i) = k;
        clusterMeans = [clusterMeans dataPt];
    else
        % set the cluster indicator to the cluster that minimizes
        % distance between d and the cluster
        z(i) = minDistIdx;
    end 
end
end














