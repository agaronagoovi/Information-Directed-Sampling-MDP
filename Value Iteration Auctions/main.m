function [ policy, paramBelief ] = main( MDPs,param,conjugatePrior )
%QLEARNING Summary of this function goes here
%   Detailed explanation goes here

trueMDP_Pssa = MDPs{9}.Pssa;


NUM_EPISODES = 50;
paramBelief = conjugatePrior;

epLength = 100;
initState = 21;

flagGoal = 1;
goalState = 1;

numActions = size(MDPs{1}.Pssa,3);
numStates = size(MDPs{1}.Pssa,2);

policy = zeros(numStates,numActions);


for e=1:NUM_EPISODES
e
paramBelief'
%%%%%Debug Code%%%%%%
% if e>=12
%     flag = 1;
% end
%%%%%%End Debug Code%%%%
    state = initState;
    if ~flagGoal
        for i=1:epLength
            [policy(state,:),action] = behaviorPolicy(MDPs,state,paramBelief,param);
            %[ action ] = eOptimalPolicy( q,state );
            [~,pssa] = observeMDP(state,action,trueMDP_Pssa);
            nextState = sample(pssa(action,:),1);
            paramBelief = bayesUpdate(MDPs,numStates,param,paramBelief,state,action,nextState);
            state=nextState;
        end
    else
        for i=1:epLength
            [policy(state,:),action] = behaviorPolicy(MDPs,state,paramBelief,param);
            %[ action ] = eOptimalPolicy( q,state );
            [~,pssa] = observeMDP(state,action,trueMDP_Pssa);
            nextState = sample(pssa(action,:),1);
            paramBelief = bayesUpdate(MDPs,numStates,param,paramBelief,state,action,nextState);
            state=nextState;
            if state == goalState
                break;
            end 
        end
    end
end

end

function posterior = bayesUpdate(MDPs,numStates,param,prior,state,action,nextState)


posterior = zeros(length(param),1);
pssa = zeros(numStates,1);

for p=1:length(param)
    %[Pssa,L] = makeMDP(param(p));
    Pssa = MDPs{p}.Pssa;
    for j=1:numStates
        pssa(j) = Pssa(state,j,action);
    end
    posterior(p) = pssa(nextState)*prior(p);
end

posterior = posterior/sum(posterior);

end

