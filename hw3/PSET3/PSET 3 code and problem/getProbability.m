% finds the probability of going to a new state given the current state
% and the desired action (stochastic actions)

function p = getProbability(s_curr, s_new, action)
if action == 1 % stay still
    
    if s_curr == s_new
        p = .9;
    else
        p = .025;
    end
    
elseif action == 2 % move 1 square right
    if (s_new(1) == s_curr(1) + 1) && (s_new(2) == s_curr(2))
        p = .9;
    elseif (s_new(1) == s_curr(1)) && (s_new(2) == s_curr(2))
        p = .925;
    else
        p = .025;
    end
    
elseif action == 3 % move 1 square up
    if (s_new(1) == s_curr(1)) && (s_new(2) == s_curr(2) + 1)
        p = .9;
    elseif (s_new(1) == s_curr(1)) && (s_new(2) == s_curr(2))
        p = .925;
    else
        p = .025;
    end
elseif action == 4 % move 1 square left
    if (s_new(1) == s_curr(1) - 1) && (s_new(2) == s_curr(2))
        p = .9;
    elseif (s_new(1) == s_curr(1)) && (s_new(2) == s_curr(2))
        p = .925;
    else
        p = .025;
    end
    
elseif action == 5 % move 1 square down
    if (s_new(1) == s_curr(1)) && (s_new(2) == s_curr(2) - 1)
        p = .9;
    elseif (s_new(1) == s_curr(1)) && (s_new(2) == s_curr(2))
        p = .925;
    else
        p = .025;
    end
    
else
    p = Inf;
    disp('choose a valid action from a = [1,2,3,4,5]')
end




























