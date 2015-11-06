function [ action,posterior ] = behaviorPolicy( MDPs,q,state,prior,DISCOUNT,param )
%BEHAVIORPOLICY Summary of this function goes here
%   Detailed explanation goes here

expRegret = zeros(size(q,2),1);
posterior = zeros(length(param),1);
jointProb = zeros(size(q,2),size(q,1),length(param));
probTran = zeros(size(q,2),size(q,1));
probOptimPolicy = prior;
for p=1:length(param)
    %[Pssa,L] = makeMDP(param(p));
    Pssa = MDPs{p,1};
    L = MDPs{p,2};
    tempOptimalValueBelief=zeros(size(q,2),1);
    policyValueBelief = zeros(size(q,2),1);
    for i=1:size(q,2)
        for j=1:size(q,1)
            pssa = Pssa(state,j,i);
            rssa = L(j);
            if pssa~=0
                tempOptimalValueBelief(i) = tempOptimalValueBelief(i) + pssa*(rssa + DISCOUNT*max(q(j,:)));
                jointProb(i,j,p) = pssa*prior(p);
                probTran(i,j) = probTran(i,j) + jointProb(i,j,p);
            end
        end
        policyValueBelief(i) = tempOptimalValueBelief(i);
    end
    optimalValueBelief = max(tempOptimalValueBelief);
    
    for i=1:size(q,2)
        regret = optimalValueBelief - policyValueBelief(i);
        expRegret(i) = expRegret(i) + prior(p)*regret;
    end
end

g = zeros(size(q,2),1);
for i=1:size(q,2)
    for j=1:size(q,1)
        for p=1:length(param)
            tempjointProb = max(jointProb(i,j,p),eps);
            probMult = max(probOptimPolicy(p)*probTran(i,j),eps);
            g(i) = g(i) + tempjointProb*log(tempjointProb/probMult);
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
pssa = zeros(size(q,1),1);

for p=1:length(param)
    %[Pssa,L] = makeMDP(param(p));
    Pssa = MDPs{p,1};
    for j=1:size(q,1)
        pssa(j) = Pssa(state,j,action);
    end
    nextState = sample(pssa,1);
    posterior(p) = pssa(nextState)*prior(p);
end

posterior = posterior/sum(posterior);


end


function [ val ] = ids(policy,g,expRegret)

if max(g)==0 && min(g) ==0
    val = (policy'*expRegret)^2;
else
    val = ((policy'*expRegret)^2)/(policy'*g);
end
end
