%% Gridworld Simulation
clear all;
close all;
clc;
%% Gridworld Parameters
N_grid = 5; % Number of grids on one direction
N_state = N_grid*N_grid; % Square Gridworld
N_act = 5; % 4 directions + null action

s_init = [1;1]; % Start in the top left corner
s_goal = [N_grid;N_grid]; % Goal is in the lower right corner

N_obstacle = 2; % Number of obstacles

obs_list = [3,4;2,2]; % Coordinates of obstacles

rew_goal = 1; % Reward for getting to the goal
rew_obs = -10; % Cost for crashing to an obstacle

noise = 0.1; % Probability of stepping in the wrong direction

params.N_grid = N_grid;
params.s_goal = s_goal;
params.rew_goal = rew_goal;
params.rew_obs = rew_obs;
params.N_act = N_act;
params.noise = noise;
params.obs_list = obs_list;
params.N_obstacle = N_obstacle;

%% Q Learning Parameters
gamma = 0.9; % Discount Factor

N_eps_length = 100; % Length of an episode
N_eps = 20; % Number of episodes
N_exec = 5; % Number f executions of the algorithm

eval_freq = 100; % How frequently current policy should be evaluated (per step)
N_eval = 10; % How many times should it be evaluated ?

cl_rate = 1.0; % Concurrent learning rate, set 0 for classical Q Learning
N_budget = 10; % Max Number of points to be kept in stack
data_method = 1; % 1 For cylic and 2 for SVD

alpha_init = 0.5; % Initial Learning Rate
alpha_dec = 0.5; % Learning Rate Decay Rate

p_eps_init = 0.2; % Initial Exploration Rate
p_eps_dec = 0.0; % Exploration Rate Decay

approximation_on=0; %0 for tabular, 1 for RBF
N_phi_s = N_state; % Number of state-features (= N_state for tabular, equal to RBF s for approx)
N_phi_sa = N_phi_s*N_act; % Number of state-action features

params.N_phi_s = N_phi_s;
params.N_phi_sa = N_phi_sa;
params.approximation_on=approximation_on;

%% Algorithm Execution

rew_exec = zeros(N_exec,1);
eval_counter = zeros(N_exec,1);

% Execution Loop
for i =1:N_exec
    %sprintf('At execution %d \n',i)
    % Reset The Q function
    theta = zeros(N_phi_sa,1);
    
    step_counter = 0;
    %eval_counter = 0;
    
    % Reset the stack matrices
    
    s_stack = zeros(2,1);% stat stack
    s_n_stack = zeros(2,1);
    u_stack= zeros(1,1);
    r_stack = zeros(1,1);
    
    F_stack = zeros(N_phi_sa,1);
    
    % Episode Loop
    for j = 1:N_eps
        fprintf('At episode %d/%d  of execution %d/%d \n',j,N_eps,i,N_exec);
        
        % Reset the initial state
        s_old = s_init;
        
        
        % Step Loop
        for k = 1: N_eps_length
            
            % Is it evlauation time ?
            
            if(mod(step_counter,eval_freq) == 0)
                %evaluate
                eval_counter(i) = eval_counter(i) + 1;
                
                rew_eval = zeros(1,N_eval);
                
                for eval_count = 1:N_eval;
                    
                    s_prev = s_init;
                    
                    for step_count = 1:N_eps_length
                        
                        action = gridworld_Q_greedy_act(theta,s_prev,params);
                        
                        s_next = gridworld_trans(s_prev,action,params);
                        
                        [rew,breaker] = gridworld_rew(s_next,params);
                        
                        rew_eval(eval_count) = rew_eval(eval_count) + rew;
                        
                        if breaker
                            break;
                        end
                        
                        s_prev = s_next;
                    end
                    
                    
                end
                
                [rew_exec(i,eval_counter(i)),~] = get_statistics(rew_eval);
                
            end
            
            % Increment the step counter
            
            step_counter = step_counter + 1;
            
            % Set the Exploration rate
            p_eps = p_eps_init/...
                (step_counter)^p_eps_dec;
            
            % Check if it is going to explore or not
            r = sample_discrete([p_eps 1-p_eps]);
            
            if r==1 % Explore
                
                p = 1/N_act.*ones(1,N_act);
                
                action = sample_discrete(p);
                
            else % Exploit
                action = gridworld_Q_greedy_act(theta,s_old,params);
            end
            
            
            % Next State
            
            s_new = gridworld_trans(s_old,action,params);
            
            % Calculate The Reward
            
            [rew,breaker] = gridworld_rew(s_new,params);
            
            % Add point to the stack
            if cl_rate>0
                if(step_counter<=N_budget)
                    
                    stack_index = step_counter;
                    
                    s_stack(:,stack_index) = s_old;
                    s_n_stack(:,stack_index) = s_new;
                    u_stack(:,stack_index) = action;
                    r_stack(:,stack_index) = rew;
                    
                    F_stack(:,stack_index) = gridworld_Q_tabular_feature(s_old,...
                        action,params);
                    
                else
                    
                    switch data_method
                        
                        case 1 % Cyclic
                            stack_index = mod(step_counter-1,N_budget) + 1;
                            
                            s_stack(:,stack_index) = s_old;
                            s_n_stack(:,stack_index) = s_new;
                            u_stack(:,stack_index) = action;
                            r_stack(:,stack_index) = rew;
                            
                        case 2 % SVM maximization
                            
                            s_min_old = min(svd(F_stack));
                            
                            s_min_vals = zeros(1,N_budget);
                            F_new_col = gridworld_Q_tabular_feature(s_old,...
                                action,params);
                            
                            for p=1:N_budget
                                
                                F =  F_stack;
                                F(:,p) = [];
                                
                                F = [F,F_new_col];
                                
                                s_min_vals(p) = min(svd(F));
                                
                            end
                            
                            [s_min_new,q] = max(s_min_vals);
                            
                            if(s_min_new > s_min_old)
                                
                                s_stack(:,q) = [];
                                s_n_stack(:,q) = [];
                                u_stack(:,q) = [];
                                r_stack(:,q) = [];
                                
                                s_stack = [s_stack,s_old];
                                s_n_stack = [s_n_stack,s_new];
                                u_stack = [u_stack,action];
                                r_stack  = [r_stack,rew];
                                
                                F_stack(:,q) = [];
                                
                                F_stack = [F_stack,F_new_col];
                                
                            end
                            
                    end
                    
                    
                    
                end
            end
            
            
            
            % Apply Concurrent - TD Update
            
            % Get the feature vector
            phi_s_old_act_old...
                = gridworld_Q_tabular_feature(s_old,action...
                ,params);
            
            % Calculate The Values
            

            val_old = gridworld_Q_value(theta,s_old,action,params);
            action_max = gridworld_Q_greedy_act(theta,s_new,params);
            val_new = gridworld_Q_value(theta,s_new,action_max,params);
            
            % Calculate Learning Rate
            alpha =  alpha_init/...
                (step_counter)^alpha_dec;
            
            
            % Calculate TD
            
            TD = (rew + gamma*val_new - val_old);
            
            % Calculate the concurrent part
            
            cc = zeros(N_phi_sa,1);
            
            [~,c_size] = size(s_stack);
            if cl_rate>0
            for m = 1:c_size
                
                s_m = s_stack(:,m);
                s_n_m = s_n_stack(:,m);
                u_m = u_stack(:,m);
                r_m = r_stack(:,m);
                
                phi_m = gridworld_Q_tabular_feature(s_m,u_m,params);
                val_m = gridworld_Q_value(theta,s_m,u_m,params);
                
                a_m =  gridworld_Q_greedy_act(theta,s_m,params);
                val_n_m = gridworld_Q_value(theta,s_n_m,a_m,params);
                
                TD_m = r_m + gamma*val_n_m - val_m;
                
                cc = cc + TD_m*phi_m;
                
            end
            end
            cc = cl_rate.*cc;
            
            % Update Theta
            
            theta = theta + alpha*(TD.*phi_s_old_act_old + cc);
            
            % Loop to new variables
            
            s_old = s_new;
            
            
            % Check Termination
            
            if breaker
                break;
            end
            
            
            
        end
        
        
        
    end
    
    
    
    
end

%% Post Process

% Find the minimum number of evals

min_eval = min(eval_counter);

rew_exec = rew_exec(:,1:min_eval);

rew_total = zeros(1,min_eval);
std_total = zeros(1,min_eval);


for m =1:min_eval
    
    [rew_total(m),std] = get_statistics(rew_exec(:,m));
    
    std_total(m) = 0.1*std;
end


%% Plots
t = 1:min_eval;
t = t.*eval_freq;
%plot(t,rew_total);
errorbar(t,rew_total,std_total);