%% Gridworld Simulation with GPs for Q learning
clear all;
close all;
clc;
%% Gridworld Parameters
N_grid = 5; % Number of grids on one direction
N_state = N_grid*N_grid; % Square Gridworld
N_act = 5; % 4 directions + null action

n_state_dim=2;
s_init = [1;1]; % Start in the bottom left corner
a_init=2;
s_goal = [N_grid;N_grid]; % Goal is in the top right corner

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
N_exec = 3; %5 Number f executions of the algorithm

eval_freq = 100; % How frequently current policy should be evaluated (per step)
N_eval = 30; % How many times should it be evaluated ?

% parameters for GPQ
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
approximation_on = 3; %0 for tabular, 1 for RBF with copy-paste features, 2 for RBF with centers over actions,3 GP
max_points = 20;%50 %max centres allowed for RBF Kernels
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
    
end

params.approximation_on=approximation_on;
%% diagnostic parameters
convergence_diagnostic_on=1;
E_pi_phi=zeros(params.N_phi_sa,params.N_phi_sa);
E_pi_phi_m=zeros(params.N_phi_sa,params.N_phi_sa);

%% Algorithm Execution - Value Iteration
rew_exec = zeros(N_exec,1);
eval_counter = zeros(N_exec,1);
numXstates = 5;
numYstates = 5;
actions = [1; 2; 3; 4; 5];
numActions = 5;
 
% initialize value function arbitrarily for s in S
V = zeros(5,5);
pi0 = zeros(5,5);
counter = 0;
eta = .001;
done = 0;
while (done == 0);
    delta = 0;
    
    % loop over all possible states
    for i = 1:numXstates
        for j = 1:numYstates
            s_curr = [i; j];
            val_curr = V(i,j);
            
            sumTerm = zeros(1, numActions);
            % take the max reward and set it to V(i,j)
            for k = 1:numActions
                % check all the new states around our current state
                action = actions(k);
                s_new = gridworld_trans(s_curr, action, params);
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                % TODO: understand the sum and taking the max over the
                % possible actions
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                % this is the reward for being in state s_new
                [reward_new, breaker] = gridworld_rew(s_new, params);
                
                % the probability that we transitioned from s_curr to s_new
                p = getProbability(s_curr, s_new, action);
                
                % the value of the point we moved to 
                v_new = V(s_new(1), s_new(2));
                sumTerm(k) = p*(reward_new + gamma*v_new);
            end
            
            
            
            [val, idx] = max(sumValues);
            V(i,j) = val;
            
            % take the action that gives max reward
            pi0(i,j) = actions(idx);
            
            delta = max(delta, abs(val_curr - V(i,j))); 
        end
    end
    
    
    if delta < eta
        done = 1;
        delta
        counter
        policy = pi0
    else
        counter = counter + 1
    end
end


%{
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
xlabel('episodes')
ylabel('reward')
%}