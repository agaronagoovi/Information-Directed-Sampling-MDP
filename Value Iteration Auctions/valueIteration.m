function [ F,scrapTh,b,Pssa ] = valueIteration( p )
%VALUEITERATION Summary of this function goes here
%   Detailed explanation goes here

%%%%%Linearly distributed private valuations with negative slope%%%%%
q = @(b) exp(-p*(1-2*b+b^2));
phi = @(b) 1- ((3*b-1)/2)*q(b) - 3/4 * sqrt(pi/p) * erf(sqrt(p)*(1-b));
optimalb = @(F,i) (1+2*(F(i+1)-F(i)))/3;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
gamma = 0.99;

h=0.01;
s = 0.005;

initInventory = 20;

totalStates = initInventory + 1;

F = zeros(totalStates,1);
scrapTh = initInventory;

flag=0;
epsilon = 0.00001;

G = zeros(initInventory+1,1);    %G(1) = 0 paper_j = j+1 for G

while ~flag
    prevF = F;
    jmax = min(initInventory,1+scrapTh);
    b = zeros(jmax,1);
    for j=1:jmax
        b(j) = optimalb(F,j);
        G(j+1) = -h*j +gamma*(phi(b(j)) + q(b(j))*F(j+1) + (1-q(b(j)))*F(j));
    end;
    deltaG = G(2);
    if s>=deltaG
        scrapTh=0;
    else
        for j=1:jmax-1
            deltaG = G(j+2)-G(j+1);
            if s>=deltaG
                scrapTh=j;
                break;
            else
                scrapTh=initInventory+1;
            end
        end
    end
    F(1)=0;
    for i=1:initInventory
        y = max(0,i-scrapTh);
        F(i+1) = s*y-h*(i-y) + G(i-y+1);
    end;
    
    if max(F-prevF)<=epsilon*(1-gamma)/(2*gamma);
        flag=1;
    end
end

actions = 0:0.1:1;  %discretize b
totalActions = length(actions) + initInventory;
Pssa = zeros(totalStates,totalStates,totalActions);

for i=2:totalStates   %state 1 -> 0 inventory
    %scrap
    for a=1:i
        inventory = i-1;
        if a<=inventory
            Pssa(i,i-a,a)=1;
        end
    end
    %min bid
    for a=initInventory+1:totalActions
        b2 = a-initInventory;
        Pssa(i,i,a) = q(actions(b2));
        Pssa(i,i-1,a) = 1 - Pssa(i,i,a);
    end
end


end

