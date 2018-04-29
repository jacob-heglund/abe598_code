function stored_data=record_data_in_stack(s_old,s_new,action,rew,stack_index,params,stored_data);
%records data in history stack
stored_data.s_stack(:,stack_index) = s_old;
stored_data.s_n_stack(:,stack_index) = s_new;
stored_data.u_stack(:,stack_index) = action;
stored_data.r_stack(:,stack_index) = rew;
stored_data.cl_learning_rate(:,stack_index)=params.cl_rate;
