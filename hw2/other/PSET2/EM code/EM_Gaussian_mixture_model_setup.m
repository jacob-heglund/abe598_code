%gaussian mixture model clusturing with EM algo
% see Tomasi's notes

clear all
%close all
%%
load gauss_mix_data

k=max(size(means));% parameteric method, so number of parameters known
N=max(size(X));
prior_mean=randn(2,5);
prior_var=vars*0+1;
rho=[0.8 0.1 0.05 0.025 0.025];
rho_post=rho;

posterior_mean=prior_mean;
posterior_var=prior_var;

%% your code here

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
for kk=1:k
plot_gaussian_ellipsoid(posterior_mean(:,kk),diag([posterior_var(kk),posterior_var(kk)]));
end
%plot(posterior_mean(1,:),posterior_mean(2,:),'*')