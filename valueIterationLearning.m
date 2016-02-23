function [ policy, paramBelief,reward,traj ] = valueIterationLearning( MDPs,param,conjugatePrior,trueMDPIdx,temperature )
%QLEARNING Summary of this function goes here
%   Detailed explanation goes here

trueMDP_Pssa = MDPs{trueMDPIdx}.Pssa;
trueMDP_Rssa = MDPs{trueMDPIdx}.Rssa;


DISCOUNT=0.90;
NUM_EPISODES = 10;
paramBelief = conjugatePrior;

epLength = 100;
initState = 21;

flagGoal = 1;
goalState = 1;

numActions = size(MDPs{1}.Pssa,3);
numStates = size(MDPs{1}.Pssa,2);

%policy = zeros(numStates,numActions);
policy = zeros(NUM_EPISODES,epLength);
traj = zeros(NUM_EPISODES,epLength);
reward = zeros(NUM_EPISODES,1);

for e=1:NUM_EPISODES
%%%%%Debug Code%%%%%%
% if e>=12
%     flag = 1;
% end
%%%%%%End Debug Code%%%%
    state = initState;
    for i=1:epLength
        traj(e,i) = state;
        [~,action] = behaviorPolicyEntropy(MDPs,state,paramBelief,DISCOUNT,param,temperature);
        %[~,action] = behaviorPolicyEntropy_finite(MDPs,state,paramBelief,i,param,temperature);
        policy(e,i) = action;
        %[policy(state,:),action] = behaviorPolicyEntropy(MDPs,state,paramBelief,DISCOUNT,param);
        %[ action ] = eOptimalPolicy( q,state );
        [~,rssa,pssa] = observeMDP(state,action,trueMDP_Pssa,trueMDP_Rssa);
        nextState = sample(pssa(action,:),1);
        %reward(i+1) = reward(i) + DISCOUNT^(i-1)*expStepReward(policy,pssa,rssa);
        reward(e) = reward(e) + DISCOUNT^(i-1)*rssa(action,nextState);
        paramBelief = bayesUpdate(MDPs,numStates,param,paramBelief,state,action,nextState);
        state=nextState;
        if state == goalState && flagGoal
            break;
        end 
    end
%     states = log(1:0.05:10);
%     reward(e) = reward(e) - exp(states(state));
    e
    paramBelief'
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

function reward = expStepReward(policy,pssa,rssa)

reward=0;

for i=1:size(pssa,1)
    for j=1:size(pssa,2)
        reward = reward + pssa(i,j)*rssa(i,j);
    end
    reward = reward + policy(i)*reward;
end
end

function [ policy,action ] = behaviorPolicyValue( MDPs,state,prior,DISCOUNT,param )
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
	Rssa = MDPs{p}.Rssa;
    v = MDPs{p}.ValueFunction;
    policyValueBelief = zeros(numActions,1);
    for i=1:numActions
        [nextState,rssa,pssa] = observeMDP(state,i,Pssa,Rssa);
        for j=1:size(nextState)
            policyValueBelief(i) = policyValueBelief(i) + pssa(i,nextState(j))*(rssa(i,nextState(j)) + DISCOUNT*v(nextState(j)));
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
options = optimoptions('fmincon','Display','off','MaxFunEvals',300000);
policy = fmincon(obj,initPolicy,A,b,Aeq,beq,lb,ub,[],options);
policy(policy<1e-5) = 0;
policy = policy/sum(policy);
action = sample(policy,1);
end


function [ val ] = ids(policy,g,expRegret)

if max(g)==0 && min(g) ==0
    val = (policy'*expRegret)^2;
else
    val = ((policy'*expRegret)^2)/(policy'*g);
end
end
