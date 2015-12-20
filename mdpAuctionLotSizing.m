function [ Pssa,Rssa ] = mdpAuctionLotSizing( p )
%MDPAUCTION Summary of this function goes here
%   Detailed explanation goes here

%%%%%Linearly distributed private valuations with negative slope%%%%%
psi = @(k,n) 1 - k/(n+1);
revenue = @(x) x*psi(x+1,n); %revenue is \pi(x,n)
phi = @(x) x*(1-poisscdf(x,p)) - (x*(x+1)*(1-poisscdf(x+1,p)))/p;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

h=0.01;
s = 0.005;


initInventory = 20;

totalStates = initInventory + 1;
totalActions = initInventory + initInventory;
Pssa = zeros(totalStates,totalStates,totalActions);
Rssa = zeros(totalStates,totalStates,totalActions);

Pssa(1,1,initInventory+1:totalActions) = 1;
for i=2:totalStates   %state 1 -> 0 inventory
    %scrap
    for a=1:i-1
        Pssa(i,i-a,a)=1;
        Rssa(i,i-a,a)=s*a-h*(i-a);
    end
    for a=i:initInventory
        Pssa(i,1,a)=1;
        Rssa(i,1,a)=-1e10;
    end
    %auction
    for a=initInventory+1:initInventory+i-1
        x = a-initInventory;
        Pssa(i,i,a) = poisscdf(x,p);
        Pssa(i,i-x,a) = 1 - Pssa(i,i,a);
        Rssa(i,i,a) = - h*i;
        Rssa(i,i-x,a) = phi(x) - h*i;
    end
    for a=initInventory+i:totalActions
        x = i-1;
        Pssa(i,i,a) = poisscdf(x,p);
        Pssa(i,i-x,a) = 1 - Pssa(i,i,a);
        Rssa(i,i,a) = -1e10;
        Rssa(i,i-x,a) = -1e10;
    end
end
        
        
end

