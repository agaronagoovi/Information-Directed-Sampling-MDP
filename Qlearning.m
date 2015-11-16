function [ q, paramBelief ] = Qlearning( MDPs,initQ,param,conjugatePrior )
%QLEARNING Summary of this function goes here
%   Detailed explanation goes here

trueMDP_Pssa = MDPs{7,1};
trueMDP_L = MDPs{7,2};


q = initQ;

NUM_EPISODES = 10000;
UPDATE_PARAM = 0.01;
DISCOUNT=0.90;

paramBelief = conjugatePrior;

epLength = 1000;
initState = 21;

flagGoal = 1;
goalState = 1;


for e=1:NUM_EPISODES
e
paramBelief'
%%%%%Debug Code%%%%%%
% if e>=12
%     flag = 1;
% end
%%%%%%End Debug Code%%%%
    %state = initState;
    state = sample(1/10*(ones(10,1)),1) + 11;
    if ~flagGoal
        for i=1:epLength
            action = behaviorPolicy(MDPs,q,state,paramBelief,DISCOUNT,param);
            %[ action ] = eOptimalPolicy( q,state );
            [~,rssa,pssa] = observeMDP(state,action,trueMDP_Pssa,trueMDP_L);
            nextState = sample(pssa(action,:),1);
            paramBelief = bayesUpdate(MDPs,size(q,1),param,paramBelief,state,action,nextState);
            reward = rssa(action,nextState);
            q(state,action) = q(state,action) + UPDATE_PARAM*(reward+DISCOUNT*max(q(nextState,:))-q(state,action));
            state=nextState;
        end
    else
        for i=1:epLength
            action = behaviorPolicy(MDPs,q,state,paramBelief,DISCOUNT,param);
            %[ action ] = eOptimalPolicy( q,state );
            [~,rssa,pssa] = observeMDP(state,action,trueMDP_Pssa,trueMDP_L);
            nextState = sample(pssa(action,:),1);
            paramBelief = bayesUpdate(MDPs,size(q,1),param,paramBelief,state,action,nextState);
            reward = rssa(action,nextState);
            q(state,action) = q(state,action) + UPDATE_PARAM*(reward+DISCOUNT*max(q(nextState,:))-q(state,action));
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
    Pssa = MDPs{p,1};
    for j=1:numStates
        pssa(j) = Pssa(state,j,action);
    end
    posterior(p) = pssa(nextState)*prior(p);
end

posterior = posterior/sum(posterior);

end

