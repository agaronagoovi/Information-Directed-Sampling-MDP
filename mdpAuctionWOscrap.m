function [ Pssa,Rssa ] = mdpAuction( p )
%MDPAUCTION Summary of this function goes here
%   Detailed explanation goes here

q = @(b) exp(-p*(1-2*b+b^2));
phi = @(b) 1- ((3*b-1)/2)*q(b) - 3/4 * sqrt(pi/p) * erf(sqrt(p)*(1-b));

actions = 0:0.1:1;  %discretize b
initInventory = 20;

totalStates = initInventory + 1;
Pssa = zeros(totalStates,totalStates,length(actions));
Rssa = zeros(totalStates,totalStates,length(actions));


Pssa(1,1,:) = 1;
for i=2:totalStates   %state 1 -> 0 inventory
    for a=1:length(actions)
        Pssa(i,i,a) = q(actions(a));
        Pssa(i,i-1,a) = 1 - Pssa(i,i,a);
        Rssa(i,i,a) = phi(actions(a));
        Rssa(i,i-1,a) = Rssa(i,i,a);
    end
end
        
        
end

