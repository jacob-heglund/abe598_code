% All values calculated using log base e (natural log).
% Calculate the entropy of each location and rank each location from highest
% entropy to the lowest entropy. 

h1 = -(.1*log(.1)+.8*log(.8)+.1*log(.1))
h2 = -(.8*log(.8)+(1/20)*log(1/20)+(3/20)*log(3/20))
h3 = -(.1*log(.1)+.7*log(.7)+.2*log(.2))
h4 = -((1/3)*log(1/3)+(1/3)*log(1/3)+(1/3)*log(1/3))
h5 = -(.4*log(.4)+.5*log(.5)+.1*log(.1))

%from least to greatest
% h2 < h1 < h3 < h5 < h4

% Suppose that you send the exploration vehicles to only the 2 locations with the
% highest entropy with respect to the pmf in the above table. Calculate the Kullback Leibler
% divergence between the 2 highest entropy locations using the
% appropriate pmfs from the below table as the models updated due to the
% observation of your 2 exploration vehicles. 

% calculate the KL Divergence between location 4 and 5 after observations
% update the probabilities.  Make sure to avoid a division by zeros!
DKL = ((1/2)*log((1/2)/(2/5)))+((1/2)*log((1/2)/(1/2)))