function [ action,nextState,reward,posterior,policy ] = behaviorPolicy( q,state,prior,policy,DISCOUNT )
%BEHAVIORPOLICY Summary of this function goes here
%   Detailed explanation goes here
param = 0:0.1:1;

expRegret = 0;
posterior = zeros(length(param),1);
jointProb = zeros(1,4);
joinProbIter=0;
probOptimPolicy = prior;
for p=1:length(param)
    [Pssa,L] = makeMDP(param(p));
    tempOptimalValueBelief=zeros(size(q,2),1);
    for i=1:size(q,2)
        [nextState,rssa,pssa] = observeMDP(state,i,Pssa,L);
        for j=1:size(nextState)
            tempOptimalValueBelief(i) = tempOptimalValueBelief + pssa(i,nextState(j))*(rssa(i,nextState(j)) + DISCOUNT*max(q(nextState(j),:)));
            joinProbIter=joinProbIter+1;
            jointProb(joinProbIter,:) = [p,i,nextState(j),pssa(i,nextState(j))*prior(p)];
        end
    end
    optimalValueBelief = max(tempOptimalValueBelief);


    policyValueBelief = 0;
    action = policy(state);
    [nextState,rssa,pssa] = observeMDP(state,action,Pssa,L);
    for j=1:size(nextState)
        policyValueBelief = policValueBelief + pssa(action,nextState(j))*(rssa(action,nextState(j)) + GAMMA*q(nextState(j),policy(nextState(j))));
    end

    regret = optimalValueBelief - policyValueBelief;
    expRegret = expRegret + prior(param(p))*regret;
end
PossNextStates = unique(jointProb(:,3));

for i=1:size(q,2)
    for j=1:length(PossNextStates)
        temp = find(jointProb(:,2)==i);
        temp2 = find(jointProb(temp,3)==j);
        idx = temp(temp2);
        probTran(i,j) = sum(jointProb(idx,4));
    end
end



    if length(nextState)>1
        nextState = sampleState(nextState,pssa(action,:));
    end
    
    posterior(p) = pssa(policy(state),nextState)*prior(p);

posterior = posterior/sum(posterior);
reward = rssa(action,nextstate);

end

