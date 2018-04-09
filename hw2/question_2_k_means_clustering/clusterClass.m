% IDEA: make all of the clusters in a single class
classdef clusterClass
    properties
        % the center of this cluster in 2D space
        centroid = zeros(2,1);
        
        % an array of data points belonging to this cluster.  Each member is a
        % column in this array.
        members = [];
    end 
%TODO: get the number of members and make it a property of the class.
% these two ways of doing that cause octave to crash, yay
    %{
    properties
    % the number of data points belonging to this cluster
        numMembers = size(clusterClass.members)(2);
    end
    methods
        function n = numMembers
            n = size(clusterClass.members)(2);
        end
    end
    %}    
end