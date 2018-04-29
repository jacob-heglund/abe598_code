function [mean,var] = get_mean_std(in)
%gets statistics
%Author: unknown

mean_old = 0;
mean = 0;
std = 0;

for k = 1:length(in)
          
    mean = mean_old + (in(k) - mean_old)/k;
       
    
    std = ((k-1)*std + (in(k) - mean)*(in(k) - mean_old))/(k);
        
    mean_old = mean;
    
end

var = sqrt(std);

end