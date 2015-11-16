function [ v,policy ] = valueIteration( Pssa,Rssa )
%VALUEITERATION Summary of this function goes here
%   Detailed explanation goes here

DISCOUNT = 0.90;

nx = size(Pssa,1);
nu = size(Pssa,3);

v = zeros(nx,1);                        % value function
policy = ones(nx,1)*round(nu+1)/2;      % policy
H = zeros(nx,nu);                       % Hamiltonian

flagGoal = 1;
goalState = 1;

tol = 1e-6;
%--------- value iteration ----------------------------------------------
while 1,
    vold = v;

    % compute Hamiltonian for current v
    for iu = 1:nu
        H(:,iu) = (Pssa(:,:,iu).*Rssa(:,:,iu))*ones(nx,1) + DISCOUNT*Pssa(:,:,iu)*v;
    end

    % update v, compute policy
    [v, policy] = max(H,[],2);

    if flagGoal,
        v(goalState) = 0;
    end

    % check for convergence
    if max(abs(v-vold))<tol,
        break;
    end
end


end


