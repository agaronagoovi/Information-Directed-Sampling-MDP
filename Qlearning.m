function [ q ] = Qlearning( initQ )
%QLEARNING Summary of this function goes here
%   Detailed explanation goes here

q = initQ;

NUM_EPISODES = 100;
UPDATE_PARAM = 1;
DISCOUNT=0.9;

epLength = 100;
initState = 1;

for e=1:NUM_EPISODES
    state = initState;
    for i=1:epLength
        action = behaviorPolicy(q,state);
        [nextState,reward] = observe(state,action);
        q(state,action) = q(state,action) + UPDATE_PARAM*(reward+DISCOUNT*max(q(nextState,:))-q(nextState,action));
        state=nextState;
    end
end

end

