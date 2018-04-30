function [Q_val_opt,action] = gridworld_Q_greedy_act(theta,s,params,gpr)
N_act = params.N_act;
act_val = zeros(1,N_act);

for l=1:N_act
    a = l;                
    act_val(l) = gridworld_Q_value(theta,s,a,params,gpr);
end
[Q_val_opt,action] = max(act_val);

end