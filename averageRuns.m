function [ reward,paramBelief,temperature ] = averageRuns( valueMDPs,param,conjugatePrior,trueMDPIdx )
%AVGRUNS Summary of this function goes here
%   Detailed explanation goes here

%temperature = [0,0.001,0.01,0.1,0.2,1,2];
temperature = 0.1;
paramBelief = zeros(length(temperature),length(param));
reward = zeros(length(temperature),10);

NUMRUNS = 100;

for k=1:length(temperature)
    for i=1:NUMRUNS
        [ ~, idsparamBelief,idsreward,~ ] = valueIterationLearning( valueMDPs,param,conjugatePrior,trueMDPIdx,temperature(k) );
        reward(k,:) = reward(k,:) + idsreward';
        paramBelief(k,:) = paramBelief(k,:) + idsparamBelief';
    end
end

reward = reward/NUMRUNS;
paramBelief = paramBelief/NUMRUNS;

end


