%========================== gp_regression =================================
%  
%  This code implements some basic Gaussian process regression code.
%  In this version of the code, we assume that the predictions are done
%  pointwise
%
%  Reference(s): 
%    C. Rasmussen and C.K Williams 
%       -Gaussian Processes for Machine Learning, Chapter 2.2, pg 19.
% 
%  Inputs:
%    data	      - n x d data matrix, with each row as a value. 
%    observations - n x 1 vector with each row as an observation
%    sigma  	  - the bandwidth for the kernel; can be a covariance 
%                   matrix or just a scalar parameter
%    noise_p      - estimated noise in observations; must be scalar 
%  Outputs:
%    gpr          - class pointer
%
%========================== gp_regression =================================
%  Name:	gp_regression.m
%
%  Author: 	Hassan A. Kingravi
%
%  Created:  01/23/2012
%  Modified: 01/23/2012
%
%========================== gp_regression =================================
function gpr = gp_regression(data,observations,sigma,noise_p)

%first compute the associated kernel matrix and coefficients
n = size(data,1);
K = kernel(data',data',sigma);
%L = chol(K + noise_p*eye(n));
%alpha = L'\(L\observations);
K = K + noise_p*eye(n);

L = inv(K);
alpha = L*observations;


gpr.learn = @learn;
gpr.get_kernel = @get_kernel;

%----------------------------- learn --------------------------------------
% Given a new vector x, output the prediction
%
%  Inputs:
%    x	          - 1 x d vector. 
%  Outputs:
%    f            - mean of the posterior
%    var_x        - variance of the posterior
%    l_marg       - the log marginal likelihood
%--------------------------------------------------------------------------
function [f,var_x,l_marg] = learn(x)
 %compute the prediction using the given formulae
 k = kernel(x',data',sigma)';

 f = k'*alpha;
 var_x = kernel(x',x',sigma) - k'*L*k;
 
%  v = L\k;
%  ls = diag(L);
%  ls = log(ls); 
%  
%  f = k'*alpha;
%  var_x = kernel(x',x',sigma)' - v'*v;
%  l_marg = -0.5*observations'*alpha - sum(ls) - (n/2)*log(2*pi);
end

function [Kr] = get_kernel()
  Kr = K;
end


%------------------------------ kernel ------------------------------------
% Helper function which computes the Gaussian kernel for a given sigma 
% between two datasets
%--------------------------------------------------------------------------
function v = kernel(x,y,sigma)
 if(length(sigma) == 1) %scalar sigma
  d=x'*y;
  dx = sum(x.^2,1);
  dy = sum(y.^2,1);
  val = repmat(dx',1,length(dy)) + repmat(dy,length(dx),1) - 2*d;
  v = exp(-val./(2*sigma^2));
 else
  isigma = inv(diag(sigma.^2));
  d =  (x'*isigma)*y;
  dx = sum((x'*isigma)'.*x,1);
  dy = sum((y'*isigma)'.*y,1);
  val = repmat(dx',1,length(dy)) + repmat(dy,length(dx),1) - 2*d;
  v = exp(-val./2);
 end
end

end