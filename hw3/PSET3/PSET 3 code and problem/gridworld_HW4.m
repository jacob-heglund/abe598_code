%% Gridworld Simulation with GPs for Q learning
clear all;
close all;
clc;
%% Gridworld Parameters
N_grid = 5; % Number of grids on one direction
N_state = N_grid*N_grid; % Square Gridworld
N_act = 5; % 4 directions + null action

n_state_dim=2;
s_init = [1;1]; % Start in the top left corner
a_init=2;
s_goal = [N_grid;N_grid]; % Goal is in the lower right corner

N_obstacle = 0; % Number of obstacles

obs_list = [];%[3,4;2,2]; % Coordinates of obstacles

rew_goal = 1; % Reward for getting to the goal
rew_obs = -10; % Cost for crashing to an obstacle

noise = 0.1; % Probability of stepping in the wrong direction

params.N_grid = N_grid;
params.s_goal = s_goal;
params.rew_goal = rew_goal;
params.rew_obs = rew_obs;
params.N_state_dim=n_state_dim;
params.N_act = N_act;
params.noise = noise;
params.obs_list = obs_list;
params.N_obstacle = N_obstacle;

%% Q Learning Parameters
gamma = 0.9; % Discount Factor
params.gamma=gamma;

N_eps_length =100; % Length of an episode
N_eps = 200; %200 Number of episodes
N_exec = 5; %5 Number f executions of the algorithm

eval_freq = 100; % How frequently current policy should be evaluated (per step)
N_eval = 30; % How many times should it be evaluated ?


N_budget = 25; % Max Number of points to be kept in history stack
params.N_budget=N_budget;
data_method = 2; % 1 For cylic and 2 for SVD
params.epsilon_data_select=0.2;
stack_index=0;
points_in_stack=0;

alpha_init = 0.5; % Initial Learning Rate
alpha_dec = 0.5; % Learning Rate Decay Rate
mu=1;
p_eps_init = 0.8; % Initial Exploration Rate
p_eps_dec = 0.1; % Exploration Rate Decay
approximation_on=0; %0 for tabular, 1 for RBF with copy-paste features, 2 for RBF with centers over actions,3 GP, 4 BKR-CL
max_points=25;%50 %max centres allowed for RBF Kernels
cl_on=0;%is concurrent learning on?
params.cl_rate = .1; % Concurrent learning rate, set 0 for classical Q Learning
cl_rate_decrease=1.1;
tol=1e-4;%1e-4;



if approximation_on==0
    params.N_phi_s = N_state; % Number of state-features (= N_state for tabular, equal to RBF s for approx)
    params.N_phi_sa = params.N_phi_s*params.N_act;
    params.approximation_on=approximation_on;
    params.state_action_slicing_on=1;
    gpr = onlineGP_RL(0,0,0,0,params);
elseif approximation_on==1
    load rbf_c_tabular
    params.N_phi_s = max(size(rbf_c))+1;% equal to RBF s for approx, 1 for bias
    
    rbf_mu = ones(params.N_phi_s,1)*mu;
    params.rbf_c=rbf_c;
    params.rbf_mu=rbf_mu;
    params.bw=1; %RBF bias
    params.N_phi_sa = params.N_phi_s*params.N_act; % Number of state-action features
    params.state_action_slicing_on=1;
    gpr = onlineGP_RL(0,0,0,0,params);
elseif approximation_on==3%GP approx
    params.N_phi_s = 1;
    params.N_phi_sa = params.N_phi_s*params.N_act; % Number of state-action features
    params.rbf_mu=2;
    bandwidth =  params.rbf_mu;
    wn=0.001;
    params.wn=wn;
    params.tol =tol;
    strg_indx=1;
    strg_counter=1;
    params.max_points=max_points;
    pp=1;
    theta=0;%since this is not relevant for GPs
    params.sparsification=1;%1=KL divergence, 2=oldest
    params.state_action_slicing_on=1;
    x_input=[s_init;a_init];
    meas_reward=0;
    meas_Qmax=0;
    gpr = onlineGP_RL(bandwidth,wn,max_points,tol,params);
    %initialize GP
    gpr.process(x_input,meas_reward,meas_Qmax,params);
    [mean_post var_post] = gpr.predict(x_input);
    var_post = 2-var_post; %slight hack
elseif approximation_on==4 %BKR-CL
    dictionary_size = max_points;
    rbf_c = zeros(2,dictionary_size);
    o = skernel(rbf_c',0.01,dictionary_size,tol);
    params.rbf_c = o.get_dictionary();%dictionary of RBF center
    params.rbf_c = params.rbf_c';%transpose
    params.N_phi_s = size(params.rbf_c,2)+1;%get n2 based on center allocation
    rbf_mu = ones(params.N_phi_s,1)*mu;
    params.rbf_mu = rbf_mu;
    params.bw=1; %RBF bias
    params.N_phi_sa = params.N_phi_s*params.N_act; % Number of state-action features
    params.state_action_slicing_on=1;
    gpr = onlineGP_RL(0,0,0,0,params);
    params.cl_on=cl_on;
    if params.cl_on==1
        stored_data.s_stack = zeros(n_state_dim,max_points);
        stored_data.s_n_stack = zeros(n_state_dim,max_points);
        stored_data.u_stack = zeros(1,max_points);
        stored_data.r_stack = zeros(1,max_points);
        stored_data.index=0;
        stored_data.points_in_stack=stored_data.index;
    end
    
end

params.approximation_on=approximation_on;

%% diagnostic parameters
convergence_diagnostic_on=1;
E_pi_phi=zeros(params.N_phi_sa,params.N_phi_sa);
E_pi_phi_m=zeros(params.N_phi_sa,params.N_phi_sa);






%% Algorithm Execution

rew_exec = zeros(N_exec,1);
eval_counter = zeros(N_exec,1);


% Execution Loop
for i =1:N_exec
    %sprintf('At execution %d \n',i)
    % Reset The Q function
    if approximation_on==1 || approximation_on==0
        theta = zeros(params.N_phi_sa,1);
        
    elseif approximation_on==3 %GP
        x_input=[s_init;a_init];
        %s_next = gridworld_trans(s_init,a_init,params);
        %[rew,breaker] = gridworld_rew(s_next,params);
        %[Q_opt,action] = gridworld_Q_greedy_act(theta,s_next,params,gpr);
        meas_reward=0;%%rew;
        meas_Qmax=0;%Q_opt;
        
        %initialize GP
        gpr.process(x_input,meas_reward,meas_Qmax,params);
        [mean_post var_post] = gpr.predict(x_input);
        var_post = 2-var_post; %slight hack
        %                params.N_phi_s=gpr.get('current_size');
        
        
    elseif approximation_on==4 %BKR-CL
        dictionary_size = max_points;
        
        rbf_c = zeros(2,dictionary_size);
        o = skernel(rbf_c',0.01,dictionary_size,tol);
        params.rbf_c = o.get_dictionary();%dictionary of RBF centers
        params.rbf_c = params.rbf_c';%transpose
        params.N_phi_s = size(params.rbf_c,2)+1;
        params.N_phi_sa=params.N_phi_s*params.N_act;
        params.n2=size(params.rbf_c,2);%get n2 based on center allocation get n2 based on center allocation
        theta = zeros(params.N_phi_sa,1);
        if params.cl_on==1
            stored_data.s_stack = zeros(n_state_dim,max_points);
            stored_data.s_n_stack = zeros(n_state_dim,max_points);
            stored_data.u_stack = zeros(1,max_points);
            stored_data.r_stack = zeros(1,max_points);
            stored_data.cl_learning_rate=zeros(1,max_points);
            stored_data.index=0;
            stored_data.points_in_stack=stored_data.index;
            %             [stored_data]=record_data_in_stack(s_init*0,s_new,a_init,0,stored_data.index,params,stored_data);
            
        end
    end
    
    step_counter = 0;
    %eval_counter = 0;
    for j = 1:N_eps
        fprintf('At episode %d/%d  of execution %d/%d \n',j,N_eps,i,N_exec);
        
        % Reset the initial state
        s_old = s_init;
        for k = 1: N_eps_length
            % Is it evlauation time ?
            
            if(mod(step_counter,eval_freq) == 0)
                %evaluate
                eval_counter(i) = eval_counter(i) + 1;
                
                rew_eval = zeros(1,N_eval);
                
                for eval_count = 1:N_eval;
                    
                    s_prev = s_init;
                    
                    for step_count = 1:N_eps_length
                        
                        [Q_opt,action] = gridworld_Q_greedy_act(theta,s_prev,params,gpr);
                        
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
            end%end eval loop
            % Increment the step counter
            
            step_counter = step_counter + 1;
            
            % Set the Exploration rate
            p_eps = p_eps_init/...
                (step_counter)^p_eps_dec;
            
            % Check if going to explore or not
            r = sample_discrete([p_eps 1-p_eps]);
            
            if r==1 % Explore
                
                p = 1/N_act.*ones(1,N_act);
                
                action = sample_discrete(p);
                
            else % Exploit
                [Q_opt,action] = gridworld_Q_greedy_act(theta,s_old,params,gpr);
            end
            
            
            
            
            % Next State
            
            s_new = gridworld_trans(s_old,action,params);
            
            % Calculate The Reward
            
            [rew,breaker] = gridworld_rew(s_new,params);
            
            
            if approximation_on==3%GP approx
                [Q_opt,action_max] = gridworld_Q_greedy_act(theta,s_new,params,gpr);
                meas_reward=rew;
                meas_Qmax=Q_opt;%+randn*0.01;%wn
                x=[s_old;action];
                gpr.update(x,meas_reward,meas_Qmax,params);
                %  params.N_phi_s=gpr.get('current_size');
            elseif approximation_on==0||approximation_on==1%normal no GP
                alpha =  alpha_init/...
                    (step_counter)^alpha_dec;
                % Get the feature vector
                phi_s_old_act_old...
                    = gridworld_Q_calculate_feature(s_old,action...
                    ,params);
                % Calculate The Values
                val_old = gridworld_Q_value(theta,s_old,action,params);
                [Q_opt,action_max] = gridworld_Q_greedy_act(theta,s_new,params,gpr);
                val_new = gridworld_Q_value(theta,s_new,action_max,params);
                TD = (rew + gamma*val_new - val_old);
                theta = theta + alpha*(TD.*phi_s_old_act_old);
                
                if convergence_diagnostic_on==1 && approximation_on==1
                    phi_max=gridworld_Q_RBF(s_new,action_max,params);%% for diagnostic
                    E_pi_phi=phi_s_old_act_old*phi_s_old_act_old'+E_pi_phi;
                    E_pi_phi_m=phi_s_old_act_old*phi_max'+E_pi_phi_m;
                end
                
                
            elseif approximation_on==4 %BKR
                alpha =  alpha_init/...
                    (step_counter)^alpha_dec;
                
                if params.cl_on==1
                    if stored_data.index==0
                        stored_data.index=1;
                        [stored_data]=record_data_in_stack(s_old,s_new,a_init,rew,stored_data.index,params,stored_data);
                    end
                end
                
                if mod(k*10,5) == 0
                    %these two elements are shifted downward
                    %send current state to add_point, decided whether or not to add the
                    %current state into the dictionary as rbf center. If flag then added
                    %and kicked out ptNr
                    [flag, ptNr, old_x] = o.add_point(s_old');
                    
                    if(flag > 0)
                        %create the new dictionary
                        old_x = old_x';
                        params.rbf_c = o.get_dictionary();
                        params.rbf_c = params.rbf_c';
                        params.N_phi_s = size(params.rbf_c,2)+1;
                        params.N_phi_sa=params.N_phi_s*params.N_act;
                        params.rbf_mu = mu*ones(params.N_phi_s+1,1);
                        new_center=1;
                        
                        
                        if ptNr > 0 && new_center==1
                            %add a state to W, or replace the old one
                            if params.cl_on==1
                                [stored_data]=record_data_in_stack(s_old,s_new,action,rew,ptNr,params,stored_data);
                            end
                        elseif ptNr==0 %or add a weight for the point that has been added
                            theta = [theta; zeros(params.N_act,1)];
                            if params.cl_on==1
                                stored_data.index=stored_data.index+1;
                                stored_data.points_in_stack=min(stored_data.index,max_points);
                                [stored_data]=record_data_in_stack(s_old,s_new,action,rew,stored_data.index,params,stored_data);
                            end
                        end
                    end
                end %end mod if
                % number_kernels(k)=size(params.rbf_c,2);
                % Get the feature vector
                phi_s_old_act_old...
                    = gridworld_Q_calculate_feature(s_old,action...
                    ,params);
                % Calculate The Values
                val_old = gridworld_Q_value(theta,s_old,action,params);
                [Q_opt,action_max] = gridworld_Q_greedy_act(theta,s_new,params,gpr);
                val_new = gridworld_Q_value(theta,s_new,action_max,params);
                phi_max=gridworld_Q_RBF(s_new,action_max,params);%% for diagnostic
                TD = (rew + gamma*val_new - val_old);
                theta = theta + alpha*(TD.*phi_s_old_act_old);
                
                if params.cl_on==1
                    cc=calculate_CLQ_concurrent_gradient(theta,params,stored_data);
                    cc = params.cl_rate.*cc;
                    theta=theta+alpha*cc;
                    stored_data.cl_learning_rate=max(stored_data.cl_learning_rate/cl_rate_decrease,0.000001);
                end
                %                 if convergence_diagnostic
                %                     E_pi_phi=phi_s_old_act_old*phi_s_old_act_old'+E_pi_phi;
                %       %             E_pi_phi_m=phi_max*phi_max'+E_pi_phi_m;
                %                 end
            end
            
            
            s_old = s_new;
            %% convergence diagnostics
            %
            
            
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