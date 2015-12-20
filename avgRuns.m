function [ reward ] = avgRuns( valueMDPs,param,conjugatePrior,trueMDPIdx )
%AVGRUNS Summary of this function goes here
%   Detailed explanation goes here

reward = zeros(100,1);
for i=1:10
    [ ~, ~,idsreward,~ ] = valueIterationLearning( valueMDPs,param,conjugatePrior,trueMDPIdx );
    reward = reward + idsreward;
end

reward = reward/10;
end


