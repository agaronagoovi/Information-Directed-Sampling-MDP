function [ nextState,rssa,pssa ] = observeMDP( state,action,Pssa,L )
%OBSERVEMDP Summary of this function goes here
% y -> number of units scrapped
% b -> minimum bid
% s-> revenue per unit of the units scrapped


nextState = [];
pssa = zeros(size(Pssa,3),size(Pssa,2));
rssa = zeros(size(Pssa,3),size(Pssa,2));
for i=1:size(Pssa,2)
    for j=1:size(Pssa,3)
        pssa(j,i) = Pssa(state,i,j);
        rssa(j,i) = L(i);
    end
    if pssa(action,i)>0
        nextState = [nextState;i];
    end
end
end

