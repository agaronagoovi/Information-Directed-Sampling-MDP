function [exp_cost, trajs, policy] = evalPolicy(q, Pssa, Rssa,policy)
%EVALPOLICY Summary of this function goes here
%   Detailed explanation goes here
nu = size(q,2);
nx = size(q,1);
alpha = 0.98;
if nargin<4
    % compute policy
    [~, policy] = max(q,[],2);
end


START = 21;
MAXSTEPS = 1000;
MAXITER = 10;
trajs = cell(MAXITER,2);
exp_cost = 0;

goalState = 1;
flagGoal = 1;

for e=1:MAXITER
    traj = zeros(MAXSTEPS+1,1);
    cost = 0;
    s = START;
    for i=1:MAXSTEPS
        traj(i) = s;
        u = policy(s);
        p = Pssa(s,:,u);
        ns = randsample(nx,1,true,p);
        cost= cost + (alpha^(i-1))*Rssa(s,ns,u);
        s = ns;
        if s==goalState && flagGoal
            break;
        end  
    end
    traj(MAXSTEPS+1) = ns;
    trajs{e,1} = traj;
    %cost = cost/MAXSTEPS;
    exp_cost = exp_cost + cost;
    trajs{e,2} = cost;
end

exp_cost = exp_cost/MAXITER;

end