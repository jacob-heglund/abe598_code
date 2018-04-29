function phi_sa = gridworld_Q_tabular_feature(s,a,params)

N_grid = params.N_grid;
N_phi_s = params.N_phi_s;
N_phi_sa = params.N_phi_sa;

s_index = sub2ind([N_grid N_grid],...
              s(1),s(2));
          
phi_s = zeros(N_phi_s,1);
phi_s(s_index) = 1;

phi_sa = zeros(N_phi_sa,1);
phi_sa(((a-1)*N_phi_s + 1):a*N_phi_s) = phi_s;
       

end