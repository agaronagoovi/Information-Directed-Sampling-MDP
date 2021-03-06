function [ Pssa,Rssa ] = mdpAuction( p )
%MDPAUCTION Summary of this function goes here
%   Detailed explanation goes here

%%%%%Linearly distributed private valuations with negative slope%%%%%
q = @(b) exp(-p*(1-2*b+b^2));
phi = @(b) 1- ((3*b-1)/2)*q(b) - 3/4 * sqrt(pi/p) * erf(sqrt(p)*(1-b));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

h=0.03;
s = 0.02;


actions = 0:0.1:1;  %discretize b
initInventory = 20;

totalStates = initInventory + 1;
totalActions = length(actions) + initInventory;
Pssa = zeros(totalStates,totalStates,totalActions);
Rssa = zeros(totalStates,totalStates,totalActions);

Pssa(1,1,initInventory+1:totalActions) = 1;
for i=2:totalStates   %state 1 -> 0 inventory
    %scrap
    for a=1:i
        inventory = i-1;
        if a<=inventory
            Pssa(i,i-a,a)=1;
            Rssa(i,i-a,a)=s*a-h*(i-a);
        end
    end
    %min bid
    for a=initInventory+1:totalActions
        b = a-initInventory;
        Pssa(i,i,a) = q(actions(b));
        Pssa(i,i-1,a) = 1 - Pssa(i,i,a);
        Rssa(i,i,a) = phi(actions(b));
        Rssa(i,i-1,a) = Rssa(i,i,a);
    end
end
        
        
end

