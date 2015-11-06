function [ action ] = eOptimalPolicy( q,state )
%EOPTIMALPOLICY Summary of this function goes here
%   Detailed explanation goes here

param = 0.01;

policy = zeros(size(q,2),1);
for i=1:size(q,2)
    policy(i)=exp(param*q(state,i));
end

policy = policy/sum(policy);

action = sample(policy,1);


end

