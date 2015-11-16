function [ action ] = eOptimalPolicy( q,state )
%EOPTIMALPOLICY Summary of this function goes here
%   Detailed explanation goes here


policy = zeros(size(q,2),1);

softmax = 1;
if softmax
    param = 1;
    for i=1:size(q,2)
        policy(i)=exp(param*q(state,i));
    end

    policy = policy/sum(policy);
else
    eps = 0.3;
    potActions = find(q(state,:));
    if isempty(potActions)
        potActions = 1:size(q,2);
    end
    [~,tempidx] = max(q(state,potActions));
    optAction = potActions(tempidx);
    policy(optAction) = 1;
    for i=size(q,2)
        policy(i) = policy(i) + eps;
    end
    policy = policy/sum(policy);
end


action = sample(policy,1);


end

