%================================ skernel =================================
%  
%  function pt = skernel(data, sigma)
%
%  This algorithm takes as input an initial set of points {x_i}_{i=1}^n.
%  When presented with a new point x_{m+1}, it performs a linear
%  independence test in the Hilbert space generated by the kernel to see if
%  the added point is sufficiently different from the current points to
%  justify inclusion in the set. If the number of points in the dataset
%  exceed MaxNr, another test is run to see which point to kick out. 
%
%  The original paper submitted this new set of points as input into a
%  support vector regressor function. We implement this for concurrent
%  learning. Currently, only the Gaussian kernel is considered. 
%
%  Reference(s): 
%    Sparse Online Model Learning for Robot Control with SVR
%     Nguyen-Tuong, Scholkopf, Peters%
% 
%  Inputs:
%    data	    - The initial dataset. Each column is an observation.
%    sigma  	- sigma for gaussian, if a vector is passed each element is
%                 taken as a sigma for the corresponding dimension. 
%    MaxNr      - The maximum number of points allowed in the dataset. 
%                 If not supplied, this is assumed to be the current size
%                 of the dataset. 
%  Outputs:
%    pt         - instantiation of the skernel class. 
%    ptNr       - index of point that was removed from the dictionary stack
%  The following interface functions are available:
%   -[K] = kernel(x,y,sigma): calculates a kernel matrix based on sigma. 
%
%================================ skernel =================================
%  Name:	skernel.m
%
%  Author: 	Hassan A. Kingravi, Girish Chowdhary
%
%  Created:  01/11/2011
%  Modified: 01/18/2011
%=============================== skernel ==================================
function pt = skernel(data,sigma,maxNr,nu)

%Check arguments.
if (nargin < 2)
 error('skernel:: - Insufficient arguments.');
end

%Initialize the variables

%% Set up the interface member functions (make code modular later).
pt.kernel = @kernel;
pt.add_point = @add_point;
pt.get_dictionary = @get_dictionary;
pt.update2 = @update2;

%% Compute the initial values: we are going to add each point, one at a time
H=(1/sigma^2)*ones(1,size(data,2));
K = [];
Kt = cell(1);
invK = cell(1);
Ktt = cell(1);
deltas = [];
m = 1; UpNr=0; 

x = data(1,:);
dictionary(1,:) = x;
ktt = kernel(x,x,H);
Ktt{m} = ktt;
deltas(m) = ktt;
invK{m} = [];
Kt{m} = [];
K = ktt; K_inv = 1/ktt;

%start going through the current dataset
n = size(data,1);
for i=2:n
 %new sample
 x = data(i,:);
 add_point(x);
 %call add point function here

end

%------------------------------ add point ---------------------------------
% Implements Algorithm 1 in the reference. Runs the independence test, and
% if it's passed, adds a point to the dictionary using Algorithms 2 or 3,
% depending on the whether the the number of points in the dictionary is
% already equal to MaxNr. 
%--------------------------------------------------------------------------
function [flag, ptNr, old_x] = add_point(x)
 %Check arguments
 flag = 0; 
 ptNr = 0;
 old_x = zeros(size(x));
  
 %ALD value for the sample
 k = kernel(x,dictionary,H);
 ktt = kernel(x,x,H);
 a = K_inv*k';
 delta = ktt - k*a;
 
 if (delta > nu) 
  %the point is relevant
  %if there's space remaining in the dictionary, simply add the point 
  if (m < maxNr) 
   flag = 1;           
   m = m+1;
   
   %save ALD values for the new data point
   Kt{m} = k;
   invK{m} = K_inv;
   Ktt{m} = ktt;
   deltas(m) = delta;
   
   %update kernel and inverse kernel matrices
   K(:,m) = k';
   K(m,:) = [k ktt];
   clear A B;
   A = [delta*K_inv + a*a'; -a'];
   B = [-a; 1];
   clear K_inv;
   K_inv = (1/delta)*[A B]; %new inverse kernel matrix
   
   %insert new point in the dictionary
   dictionary(m,:) = x;
   
   %update ALD values for other dictionary points
   knn = kernel(x,x,H);
   
   for dataNr=1:m-1
    %dataNr 
    xt = dictionary(dataNr,:);
    D  = SetDiff(dictionary(1:end-1,:),dataNr); % Select data set without x
    kn = kernel(x,D,H);
    knt = kernel(x,xt,H);
    
    alpha_n = invK{dataNr}*kn';
    if (isempty(alpha_n))
     gamma_n = knn;
     a_t = knt;
    else
     gamma_n = knn-kn*alpha_n;
     a_t = knt-Kt{dataNr}*alpha_n;
    end
        
    
    A_t = (gamma_n*invK{dataNr} + alpha_n*alpha_n')*Kt{dataNr}' - knt*alpha_n;
    
    %Compute update
    At             = (1/gamma_n)*[A_t; a_t];
    Kt{dataNr}     = [Kt{dataNr} knt];
        
    deltas(dataNr) = Ktt{dataNr} - Kt{dataNr}*At;
    K1             = gamma_n*invK{dataNr} + alpha_n*alpha_n';
    K2             = [K1; -alpha_n'];
    invK{dataNr}   = (1/gamma_n)*[K2 [-alpha_n ; 1]];

   end
  else 
   %if there's no space remaining, add a point, and delete the one with the
   %lowest delta value
   flag = 2;
   UpNr = UpNr+1;      

   %find dictionary point with minimal delta value (ALD value)
   [val,ptNr] = min(deltas);

   %replace the deleted point by the new data point
   old_x = dictionary(ptNr,:);
   dictionary(ptNr,:)= x;
   
   %save ALD values for this new data point
   ktt = kernel(x,x,H);
   k   = kernel(x,SetDiff(dictionary,ptNr),H);
   Kt{ptNr}  = k;
   Ktt{ptNr} = ktt;
   deltas(ptNr)= ktt-k*invK{ptNr}*k';

   %update cov. and inverse-cov. matrices
   k_old  = K(ptNr,:);
   k      = kernel(x,dictionary,H);
   K(ptNr,:) = k;  % new K matrix for training
   K(:,ptNr) = k';
   d      = k-k_old;
   d(ptNr)= 0.5*d(ptNr);
   A_t    = K_inv - (1/(1+d*K_inv(:,ptNr))) * K_inv*d'*K_inv(:,ptNr)';
   K_inv  = A_t - (1/(1+d*A_t(:,ptNr))) * A_t(:,ptNr)*d*A_t; %new inverse  

   %update ALD values for other data points
    for dataNr=1:m
     if (dataNr ~= ptNr)
      xt = dictionary(dataNr,:);

      D    = SetDiff(dictionary,dataNr); % Select data set
      kn   = kernel(x,D,H);
      knt  = kernel(xt,x,H);
      if (dataNr==1) 
       k_up = k_old(2:end);
      else
       k_up = k_old([1:dataNr-1 dataNr+1:end]);
      end

      % Change index for updating matrix
      if(dataNr < ptNr) 
       upNr = ptNr-1;
      elseif (dataNr > ptNr) 
       upNr = ptNr;
      else display('Wrong Nr !!'); break;
      end

      d = kn-k_up;
      d(upNr) = 0.5*d(upNr);

      A_t = invK{dataNr} - (1/(1+d*invK{dataNr}(:,upNr))) ...
             * invK{dataNr}*d'*invK{dataNr}(:,upNr)';
      invK{dataNr} = A_t - (1/(1+d*A_t(:,upNr))) ...
                      * A_t(:,upNr)*d*A_t;

      % Compute Update
      Kt{dataNr}(upNr)= knt;
      At              = invK{dataNr}*Kt{dataNr}';
      deltas(dataNr)    = Ktt{dataNr}-Kt{dataNr}*At;

     end

   end   
  end  
 end
 
 %dictionary
end

%------------------------------ update1 -----------------------------------
% Implements Algorithm 2 in the reference, which updates the dictionary by
% adding without deleting any points in the dictionary.
% We define n := m+1 (see paper).
%--------------------------------------------------------------------------
function [dict] = get_dictionary()
 dict = dictionary; 
end

%------------------------------ update2 -----------------------------------
% Implements Algorithm 3 in the reference, which updates the dictionary by
% adding a point, and deleting a point in the dictionary.
%--------------------------------------------------------------------------
function update2()
end

%------------------------------- kernel -----------------------------------
% function v =  kernel(x,y,sigma)
%  if(length(sigma) == 1) %same sigma
%   d=x'*y;
%   dx = sum(x.^2,1);
%   dy = sum(y.^2,1);
%   val = repmat(dx',1,length(dy)) + repmat(dy,length(dx),1) - 2*d;
%   v = exp(-val./(2*sigma^2));
%  else
%   isigma = inv(diag(sigma.^2));
%   d =  (x'*isigma)*y;
%   dx = sum((x'*isigma)'.*x,1);
%   dy = sum((y'*isigma)'.*y,1);
%   val = repmat(dx',1,length(dy)) + repmat(dy,length(dx),1) - 2*d;
%   v = exp(-val./2);
%  end
%  v = v';
% end %end function
function f= kernel(x,X,H)
% Compute sqr-exponential kernel
f=[];
s=size(X,1);
for z=1:s
    dist = (x-X(z,:)).^2*H';
    f(z) = exp(-0.5*dist);
end
end

end %end class
%=============================== skernel =================================

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Return the set 'X' without the point 'dataNr' 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function f=SetDiff(X,dataNr)
s=size(X,1);
S=[]; k=1;
for m=1:s
    if (m~=dataNr)
        S(k,:)=X(m,:);
        k=k+1;
    end
end
f=S;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Computing Exponential (Gaussian) Kernel 
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
