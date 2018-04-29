function  [stored_data]=use_svm_maximizer(s_old,s_new,action,rew,params,stored_data)

stored_data.s_min_old = min(svd(stored_data.F_stack));

s_min_vals = zeros(1,params.N_budget);
F_new_col = gridworld_Q_calculate_feature(s_old,...
    action,params);

for p=1:params.N_budget
    
    F =  stored_data.F_stack;
    F(:,p) = F_new_col;%[];
    s_min_vals(p) = min(svd(F));
    
end

[stored_data.s_min_new,q] = max(s_min_vals);

if(stored_data.s_min_new > stored_data.s_min_old)
%    stored_data= record_data_in_stack(s_old,s_new,action,rew,q,params,stored_data);
    stored_data.s_stack(:,q) = s_old;
    stored_data.s_n_stack(:,q) = s_new;
    stored_data.u_stack(:,q) = action;
    stored_data.r_stack(:,q) = rew;
    stored_data.F_stack(:,q)=F_new_col;
end