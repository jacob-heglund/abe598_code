%gaussian mixture model clusturing with EM algo
% see Tomasi's notes
clear all;
clc;
close all;

%%
load gauss_mix_data

k = max(size(means)); % parameteric method, so number of parameters known
N = max(size(X)); % number of data points
data = [X'; Y'];

prior_mean=randn(2,5);
prior_var = vars*0+1;
rho = [0.8 0.1 0.05 0.025 0.025];
rho_post = rho;

% are we supposed to do something with these variables?
posterior_mean = prior_mean;
posterior_var = prior_var;

%% your code here

% step 1 - randomly initialize the centroid positions in the space (2D plane)
centroid = randn(2,5);
objarray(k,1) = clusterClass;

for i = 1:k
    objarray(i).centroid = centroid(:,i);
end

done = 0;
counter = 0;
error = 5;
tic
%while (done == 0)
for A = 1:500
    % step 2 - find the distance to each centroid from each data point
    % assign a data point as a member to the nearest centroid
    for i = 1:N
        distances = zeros(k,1);
        
        for j = 1:k
            distances(j) = norm(data(:,i) - objarray(j).centroid);
        end
        [val , index] = min(distances);
        objarray(index).members = [objarray(index).members data(:,i)];       
    end

    % step 3 - for each cluster, update the centroid position by setting it as the
    % mean position of the members in the cluster    
    for i = 1:k
        objarray(i).centroid = [mean(objarray(i).members(1,:));
                                mean(objarray(i).members(2,:))];
    end
        
    %objarray(2).centroid

    % Repeat steps 2 and 3 if the centroid positions change betweens
    % steps
    deltaMean = 0;
    
    for i = 1:k
        deltaMean = deltaMean + norm(objarray(i).oldCentroid - objarray(i).centroid, 2);
    end    
    
    if deltaMean < error
        done = 1;
    else
        objarray(i).oldCentroid = objarray(i).centroid;
        objarray(i).members = [];
    end
    
counter = counter + 1;
end
elapsedTime = toc

% oh I guess we are supposed to use those variables
for i = 1:k
    posterior_mean(:,i) = objarray(i).centroid;
end

% repeat steps 2 and 3 until no members change clusters


%% draw from posterior
%for jj = 1:max(size(means))
%    for ii = 1:30
%        y(ii,jj)=randn*posterior_var(jj)+posterior_mean(1,jj);
%        x(ii,jj)=randn*posterior_var(jj)+posterior_mean(2,jj);
%    end
%end

figure
plot(x(:,1),y(:,1),'or')
hold on
plot(x(:,2),y(:,2),'og')
hold on
plot(x(:,3),y(:,3),'om')
hold on
plot(x(:,4),y(:,4),'ok')
hold on
plot(x(:,5),y(:,5),'oc')
% x_trial=-5:1:10;
% for ii=1:max(size(x_trial))
%     y_trial(ii)=gaussian_dist([x_trial(ii);x_trial(ii)],posterior_mean(1),posterior_var(1));
% end
%  plot(x_trial,y_trial)

%% this plots the posterior mean and variance
% if your inference works, you should be able to see the right allocation
for kk = 1:k
    plot_gaussian_ellipsoid(posterior_mean(:,kk),diag([posterior_var(kk),posterior_var(kk)]));
end
%plot(posterior_mean(1,:),posterior_mean(2,:),'*')















