function [ policy,action ] = behaviorPolicyPrior( MDPs,state,prior,DISCOUNT,param )
%BEHAVIORPOLICYPRIOR Summary of this function goes here
%   Policy generated based on the prior for parameter of MDP using
%   maximization of the bellman transition paramter

numActions = size(MDPs{1}.Pssa,3);
policy = zeros(numActions,1);
for p=1:length(param)
    Pssa = MDPs{p}.Pssa;
	Rssa = MDPs{p}.Rssa;
    v = MDPs{p}.ValueFunction;
    q = zeros(numActions,1);
    for a=1:numActions
        for ns = 1:size(Pssa,1)
            q(a) = q(a) + Pssa(state,ns,a).*(Rssa(state,ns,a) + DISCOUNT*v(ns));
        end
    end
    [~,action] = max(q);
    policy(action) = policy(action) + prior(p);
end

action = sample(policy,1);

end

