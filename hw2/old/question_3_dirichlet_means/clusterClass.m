% a single cluster object
classdef clusterClass
    properties
        % the center of this cluster in 2D space
        centroid = zeros(2,1);
        
        % the center of this cluster in 2D space for the previous step
        oldCentroid = ones(2,1);
        
        % an array of data points belonging to this cluster.  Each member is a
        % column in this array.
        members = [];
        
     
    end 
end

















