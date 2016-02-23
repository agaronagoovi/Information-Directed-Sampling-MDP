function [ policy,action ] = behaviorPolicyEntropy_finite( MDPs,state,prior,step,param,k )
%BEHAVIORPOLICY Summary of this function goes here
%   Detailed explanation goes here

numActions = size(MDPs{1}.Pssa,3);
tempjointProb = zeros(1,4);
joinProbIter=0;
probOptimPolicy = prior;
H=zeros(numActions,1);
for p=1:length(param)
    %[Pssa,L] = makeMDP(param(p));
    Pssa = MDPs{p}.Pssa;
	Rssa = MDPs{p}.Rssa;
    v = MDPs{p}.ValueFunction;
    policyValueBelief = zeros(numActions,1);
    for i=1:numActions
        [nextState,rssa,pssa] = observeMDP(state,i,Pssa,Rssa);
        for j=1:size(nextState)
            policyValueBelief(i) = policyValueBelief(i) + pssa(i,nextState(j))*(rssa(i,nextState(j)) + v(nextState(j),step+1));
            joinProbIter=joinProbIter+1;
            tempjointProb(joinProbIter,:) = [p,i,nextState(j),pssa(i,nextState(j))*prior(p)];
        end
        H(i,1) = H(i,1) + prior(p)*policyValueBelief(i);
    end
end
PossNextStates = unique(tempjointProb(:,3));
probTran = zeros(numActions,length(PossNextStates));
g = zeros(numActions,1);
for i=1:numActions
    temp = find(tempjointProb(:,2)==i);       %next states index for given action
    for j=1:length(PossNextStates)
        temp2 = tempjointProb(temp,3)==PossNextStates(j);     %index for given next state and action
        idx = temp(temp2);
        if isempty(idx)
            continue;
        end
        probTran(i,j) = sum(tempjointProb(idx,4));
        for p=1:length(param)
            idx2 = idx(tempjointProb(idx,1)==p);
            if isempty(idx2)
                continue;
            end
            jointProb = max(tempjointProb(idx2,4),eps);
            probMult = max(probOptimPolicy(p)*probTran(i,j),eps);
            g(i) = g(i) + jointProb*log(jointProb/probMult);
        end
    end
end
            
H = H + k*g;
[~,action] = max(H);
policy = zeros(numActions,1);
policy(action)=1;
end