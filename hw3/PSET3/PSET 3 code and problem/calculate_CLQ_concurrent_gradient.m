function cc= calculate_CLQ_concurrent_gradient(theta,params,stored_data)
%calculates the concurrent gradient update for CLQ learning
cc = zeros(params.N_phi_sa,1);

for m = 1:stored_data.points_in_stack
%     if abs(stored_data.cl_learning_rate)>=0.0001;
    s_m = stored_data.s_stack(:,m);
    s_n_m = stored_data.s_n_stack(:,m);
    u_m = stored_data.u_stack(:,m);
    r_m = stored_data.r_stack(:,m);
    
    phi_m = gridworld_Q_calculate_feature(s_m,u_m,params);
    val_m = gridworld_Q_value(theta,s_m,u_m,params);
    
    [Q_opt,a_m] =  gridworld_Q_greedy_act(theta,s_m,params,0);
    val_n_m = gridworld_Q_value(theta,s_n_m,a_m,params);
    
    TD_m = r_m + params.gamma*val_n_m - val_m;
    
    cc = cc + stored_data.cl_learning_rate(m)*TD_m*phi_m;
%     end
end