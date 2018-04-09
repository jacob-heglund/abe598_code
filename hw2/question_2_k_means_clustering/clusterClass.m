classdef clusterClass
    properties
        % the position of this cluster in 2D space
        position;
        
        % an array of data points belonging to this cluster.  Each member is a
        % column in this array.
        members = [];
    end 
    
    properties
    % the number of data points belonging to this cluster
        numMembers = size(clusterClass.members)(2);
    end
    methods
        % here's where my functions would go, if I had any
    end
end