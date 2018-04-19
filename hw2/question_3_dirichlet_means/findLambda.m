% inputs: 
% x, y - 2d dataset with the same number of points in each array
% k_approx - a guess of the number of clusters in the dataset
 
% outputs: 
% lambda - a measure of distance in the dataset, if a datapoint is greater than
% lambda, it is part of a new cluster

function lambda = findLambda(x,y, k_approx)
data = [x; y];
numData = size(x,2);
% initialize a set T with the global mean
xAvg = mean(x);
yAvg = mean(y);
T = [xAvg; yAvg];

% for k_approx iterations, find the furthest point from the mean of set T
% and add it to the set T
for i = 1:k_approx
    % find the mean of set T as a point in the 2D plane
    xtAvg = mean(T(1,:));
    ytAvg = mean(T(2,:));
    tAvg = [xtAvg; ytAvg];
    
    % find the furthest point from the mean of set T
    dist = zeros(1, numData);
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





