function [ q, paramBelief ] = Qlearning( MDPs,initQ,param,conjugatePrior )
%QLEARNING Summary of this function goes here
%   Detailed explanation goes here

trueMDP_Pssa = MDPs{5,1};
trueMDP_L = MDPs{5,2};


q = initQ;

NUM_EPISODES = 10;
UPDATE_PARAM = 0.01;
DISCOUNT=0.98;

paramBelief = conjugatePrior;

epLength = 10000;
initState = 1;



for e=1:NUM_EPISODES
e
paramBelief'
%%%%%Debug Code%%%%%%
% if e>=12
%     flag = 1;
% end
%%%%%%End Debug Code%%%%
    state = initState;
    for i=1:epLength

        [action,paramBelief] = behaviorPolicy(MDPs,q,state,paramBelief,DISCOUNT,param);
        %[ action ] = eOptimalPolicy( q,state );
        [~,rssa,pssa] = observeMDP(state,action,trueMDP_Pssa,trueMDP_L);
        nextState = sample(pssa(action,:),1);
        reward = rssa(action,nextState);
        q(state,action) = q(state,action) + UPDATE_PARAM*(reward+DISCOUNT*max(q(nextState,:))-q(state,action));
        state=nextState;
    end
end

end

