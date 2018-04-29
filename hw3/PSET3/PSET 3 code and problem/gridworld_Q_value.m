function val = gridworld_Q_value(theta,s,a,params,gpr)

if params.approximation_on==0
    phi_sa = gridworld_Q_tabular_feature(s,a,params);
    val = theta'*phi_sa;
elseif params.approximation_on==1 || params.approximation_on==2 || params.approximation_on==4
    phi_sa= gridworld_Q_RBF(s,a,params);
    val = theta'*phi_sa;
elseif params.approximation_on==3 
    x=[s;a];
    [mean_post var_post] = gpr.predict(x);      
     val = mean_post;
end



end