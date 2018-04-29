function phi_sa=gridworld_Q_calculate_feature(s,a,params)
%call the appropriate feature function

if params.approximation_on==0
    phi_sa = gridworld_Q_tabular_feature(s,a,params);
elseif params.approximation_on==1 || params.approximation_on==2 || params.approximation_on==4
    phi_sa= gridworld_Q_RBF(s,a,params);
end