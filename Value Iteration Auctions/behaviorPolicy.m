function [ policy,action ] = behaviorPolicy( MDPs,state,prior,param )
%BEHAVIORPOLICY Summary of this function goes here
%   Detailed explanation goes here

numActions = size(MDPs{1}.Pssa,3);
expRegret = zeros(numActions,1);
tempjointProb = zeros(1,4);
joinProbIter=0;
probOptimPolicy = prior;
for p=1:length(param)
    %[Pssa,L] = makeMDP(param(p));
    Pssa = MDPs{p}.Pssa;
    v = MDPs{p}.ValueFunction;
    policyValueBelief = zeros(numActions,1);
    for i=1:numActions
        [nextState,pssa] = observeMDP(state,i,Pssa);
        for j=1:size(nextState)
            policyValueBelief(i) = policyValueBelief(i) + pssa(i,nextState(j))*v(nextState(j));
            joinProbIter=joinProbIter+1;
            tempjointProb(joinProbIter,:) = [p,i,nextState(j),pssa(i,nextState(j))*prior(p)];
        end
    end
    optimalValueBelief = max(policyValueBelief);
    
    for i=1:numActions
        regret = optimalValueBelief - policyValueBelief(i);
        expRegret(i) = expRegret(i) + prior(p)*regret;
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
            
obj = @(policy) ids(policy,g,expRegret);
Aeq = ones(1,numActions);
beq = 1;
A=[];
b=[];
lb=zeros(numActions,1);
ub=ones(numActions,1);
initPolicy = 1/numActions * ones(numActions,1);
options = optimoptions('fmincon','Display','off');
policy = fmincon(obj,initPolicy,A,b,Aeq,beq,lb,ub,[],options);
action = sample(policy,1);
end


function [ val ] = ids(policy,g,expRegret)

if max(g)==0 && min(g) ==0
    val = (policy'*expRegret)^2;
else
    val = ((policy'*expRegret)^2)/(policy'*g)^2;
end
end
