function [ action,nextState,reward,posterior ] = behaviorPolicy( q,state,prior,DISCOUNT )
%BEHAVIORPOLICY Summary of this function goes here
%   Detailed explanation goes here
param = 0:0.1:1;

expRegret = zeros(size(q,2),1);
posterior = zeros(length(param),1);
tempjointProb = zeros(1,4);
joinProbIter=0;
probOptimPolicy = prior;
for p=1:length(param)
    [Pssa,L] = makeMDP(param(p));
    tempOptimalValueBelief=zeros(size(q,2),1);
    policyValueBelief = zeros(size(q,2),1);
    for i=1:size(q,2)
        [nextState,rssa,pssa] = observeMDP(state,i,Pssa,L);
        for j=1:size(nextState)
            tempOptimalValueBelief(i) = tempOptimalValueBelief + pssa(i,nextState(j))*(rssa(i,nextState(j)) + DISCOUNT*max(q(nextState(j),:)));
            joinProbIter=joinProbIter+1;
            tempjointProb(joinProbIter,:) = [p,i,nextState(j),pssa(i,nextState(j))*size(q,2)*prior(p)];
        end
        policyValueBelief(i) = tempOptimalValueBelief(i);
    end
    optimalValueBelief = max(tempOptimalValueBelief);
    
    for i=1:size(q,2)
        regret = optimalValueBelief - policyValueBelief(i);
        expRegret(i) = expRegret(i) + prior(param(p))*regret;
    end
end
PossNextStates = unique(tempjointProb(:,3));
probTran = zeros(size(q,2),length(PossNextStates));
g = zeros(size(q,2),1);
for i=1:size(q,2)
    temp = find(tempjointProb(:,2)==i);       %next states index for given action
    for j=1:length(PossNextStates)
        temp2 = tempjointProb(temp,3)==j;     %index for given next state and action
        idx = temp(temp2);                
        probTran(i,j) = sum(tempjointProb(idx,4));
        for p=1:length(param)
            idx2 = idx(tempjointProb(idx,1)==p);
            jointProb = tempjointProb(idx2,4);
            probMult = probOptimPolicy(p)*probTran(i,j);
            g(i) = g(i) + jointProb(k)*log(jointProb(k)/probMult(k));
        end
    end
end
            
obj = @(policy) ((policy'*expRegret)^2)/policy'*g;
Aeq = ones(1,size(q,2));
beq = 1;
A=[];
b=[];
lb=zeros(size(q,2),1);
ub=ones(size(q,2),1);
initPolicy = 1/size(q,2) * ones(size(q,2),1);
policy = fmincon(obj,initPolicy,A,b,Aeq,beq,lb,ub);
action = sample(1:size(q,2),policy);

reward=zeros(length(param,1));
for p=1:length(param)
    [Pssa,L] = makeMDP(param(p));
    [nextState,rssa,pssa] = observeMDP(state,action,Pssa,L);
    if length(nextState)>1
        nextState = sample(nextState,pssa(action,:));
    end
    posterior(p) = pssa(policy(state),nextState)*prior(p);
    reward(p) = rssa(policy(state),nextState);
end

posterior = posterior/sum(posterior);
reward = posterior'*reward;
end

