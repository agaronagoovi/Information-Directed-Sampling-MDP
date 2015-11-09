function [ action ] = behaviorPolicy( MDPs,q,state,prior,DISCOUNT,param )
%BEHAVIORPOLICY Summary of this function goes here
%   Detailed explanation goes here

expRegret = zeros(size(q,2),1);
tempjointProb = zeros(1,4);
joinProbIter=0;
probOptimPolicy = prior;
for p=1:length(param)
    %[Pssa,L] = makeMDP(param(p));
    Pssa = MDPs{p,1};
    Rssa = MDPs{p,2};
    tempOptimalValueBelief=zeros(size(q,2),1);
    policyValueBelief = zeros(size(q,2),1);
    for i=1:size(q,2)
        [nextState,rssa,pssa] = observeMDP(state,i,Pssa,Rssa);
        for j=1:size(nextState)
            tempOptimalValueBelief(i) = tempOptimalValueBelief(i) + pssa(i,nextState(j))*(rssa(i,nextState(j)) + DISCOUNT*max(q(nextState(j),:)));
            joinProbIter=joinProbIter+1;
            tempjointProb(joinProbIter,:) = [p,i,nextState(j),pssa(i,nextState(j))*prior(p)];
        end
        policyValueBelief(i) = tempOptimalValueBelief(i);
    end
    optimalValueBelief = max(tempOptimalValueBelief);
    
    for i=1:size(q,2)
        regret = optimalValueBelief - policyValueBelief(i);
        expRegret(i) = expRegret(i) + prior(p)*regret;
    end
end
PossNextStates = unique(tempjointProb(:,3));
probTran = zeros(size(q,2),length(PossNextStates));
g = zeros(size(q,2),1);
for i=1:size(q,2)
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
Aeq = ones(1,size(q,2));
beq = 1;
A=[];
b=[];
lb=zeros(size(q,2),1);
ub=ones(size(q,2),1);
initPolicy = 1/size(q,2) * ones(size(q,2),1);
options = optimoptions('fmincon','Display','off');
policy = fmincon(obj,initPolicy,A,b,Aeq,beq,lb,ub,[],options);
action = sample(policy,1);
end


function [ val ] = ids(policy,g,expRegret)

if max(g)==0 && min(g) ==0
    val = (policy'*expRegret)^2;
else
    val = ((policy'*expRegret)^2)/(policy'*g);
end
end
