function [ reward ] = averageRuns( MDPs,param,conjugatePrior,trueMDPIdx )
%AVERAGERUNS Summary of this function goes here
%   Detailed explanation goes here
runs = 10;

reward = 0;

for i=1:runs
    [ ~, ~,temp,~ ] = valueIterationLearning( MDPs,param,conjugatePrior,trueMDPIdx );
    reward = reward + temp;
end

reward = reward/runs;
end

