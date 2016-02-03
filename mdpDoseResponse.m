function [ Pssa,Rssa ] = mdpDoseResponse( p )
%MDPDOSERESPONSE Summary of this function goes here
%   p -> [5,50] acc to paper
% states log(DAS), actions dosage

states = log(1:0.05:10);
actions = 0:1:10;

numStates = length(states);
numActions = length(actions);

noise = -3:0.1:3;
noiseStd = 0.05;
c = 0.028557;

Pssa = zeros(numStates,numStates,numActions);
Rssa = -10000*ones(numStates,numStates,numActions);

for i=1:numStates
    currentState = states(i);
    for j = 1:numActions
        dose = actions(j);
        nextState = currentState + log(p) - log(p+dose);
        for k=1:length(noise);
            noiseNextState = nextState + noiseStd*noise(k);
            [~,idx] = min(abs(states-noiseNextState));
            Pssa(i,idx,j) = Pssa(i,idx,j) + normpdf(noiseNextState,nextState,noiseStd);
            Rssa(i,idx,j) = -exp(states(idx))-c*dose;
        end
        Pssa(i,:,j) = Pssa(i,:,j)/sum(Pssa(i,:,j));
    end
end



end

