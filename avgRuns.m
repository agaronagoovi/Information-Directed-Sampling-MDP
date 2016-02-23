function [ reward,paramBelief ] = avgRuns( valueMDPs,param,conjugatePrior,trueMDPIdx )
%AVGRUNS Summary of this function goes here
%   Detailed explanation goes here

temperature = (1:10).^3;
paramBelief = zeros(length(temperature),length(param));
reward = zeros(length(temperature),1);


for k=1:length(temperature)
    for i=1:100
        [ ~, idsparamBelief,idsreward,~ ] = valueIterationLearning( valueMDPs,param,conjugatePrior,trueMDPIdx,temperature(k) );
        reward(k,:) = reward(k,:) + idsreward';
        paramBelief(k,:) = paramBelief(k,:) + idsparamBelief';
    end
end

reward = reward/100;
paramBelief = paramBelief/100;

end


