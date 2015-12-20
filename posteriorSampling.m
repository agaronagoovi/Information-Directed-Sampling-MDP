function [ policy, paramBelief,reward,traj ] = posteriorSampling( MDPs,param,conjugatePrior,trueMDPIdx )
%QLEARNING Summary of this function goes here
%   Detailed explanation goes here

trueMDP_Pssa = MDPs{trueMDPIdx}.Pssa;
trueMDP_Rssa = MDPs{trueMDPIdx}.Rssa;


DISCOUNT=0.90;
NUM_EPISODES = 100;
paramBelief = conjugatePrior;

epLength = 100;
initState = 21;

flagGoal = 1;
goalState = 1;

numActions = size(MDPs{1}.Pssa,3);
numStates = size(MDPs{1}.Pssa,2);

policy = zeros(numStates,numActions);
traj = zeros(NUM_EPISODES,epLength);
reward = zeros(NUM_EPISODES,1);

for e=1:NUM_EPISODES
e
paramBelief'
%%%%%Debug Code%%%%%%
% if e>=12
%     flag = 1;
% end
%%%%%%End Debug Code%%%%
    state = initState;
    sampledParam = sample(paramBelief,1);
    tempPolicy = MDPs{sampledParam}.policy;
    for i=1:epLength
        traj(e,i) = state;
        %sampledParam = sample(paramBelief,1);
        %tempPolicy = MDPs{sampledParam}.policy;
        policy(state,tempPolicy(state)) = policy(state,tempPolicy(state)) + 1;
        action = tempPolicy(state);
        
        [~,trueMDP_rssa,trueMDP_pssa] = observeMDP(state,action,trueMDP_Pssa,trueMDP_Rssa);
        nextState = sample(trueMDP_pssa(action,:),1);
        reward(e) = reward(e) + DISCOUNT^(i-1)*trueMDP_rssa(action,nextState);
        paramBelief = bayesUpdate(MDPs,numStates,param,paramBelief,state,action,nextState);
        state=nextState;
        if state == goalState && flagGoal
            break;
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
