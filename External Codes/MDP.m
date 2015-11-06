% Illustration of value iteration, policy iteration,
% and linear programming approach to solving an MDP.
%
% Emanuel Todorov, UW 2010


function [v, policy] = MDP

%--------- user-defined parameters --------------------------------------
alpha = 0.9;                            % discount factor
tol = 1E-10;                            % convergence threshold

% flags specifying problem formulation (only one can be set)
discounted = 1;
average = 0;
firstexit = 0;

% flags specifying algorithm (only one can be set)
valueiteration = 1;
policyiteration = 0;
linearprogramming = 0;

% set alpha=1 if not discounted
if ~discounted,
    alpha = 1;
end


%-------- make MDP, allocate results ------------------------------------
[P, L] = makeMDP;
nx = 2;
nu = 2;

% allocate results
v = zeros(nx,1);                        % value function
policy = ones(nx,1)*round(nu+1)/2;      % policy
H = zeros(nx,nu);                       % Hamiltonian
PP = zeros(nx,nx);                      % policy-specific transitions
LL = zeros(nx,1);                       % policy-specific costs



%--------- value iteration ----------------------------------------------
if valueiteration,

    while 1,
        vold = v;

        % compute Hamiltonian for current v
        for iu = 1:nu
            H(:,iu) = L + alpha*P(:,:,iu)*v;
        end

        % update v, compute policy
        [v, policy] = max(H,[],2);
        if average,
            c = mean(v);
            v = v - c;
            
        elseif firstexit,
            v(isgoal) = 0;
        end

        % check for convergence
        if max(abs(v-vold))<tol,
            break;
        end
    end


%--------- policy iteration ---------------------------------------------
elseif policyiteration,

    while 1,
        vold = v;
        
        % construct transitions and cost for current policy
        for ix = 1:nx
            PP(ix,:) = P(ix,:,policy(ix));
            LL(ix) = L(ix,policy(ix));
        end

        % evaluate current policy
        if discounted,
            v = (eye(nx)-alpha*PP)\LL;
            
        elseif average,
            tmp = [eye(nx)-PP, ones(nx,1); ones(1,nx) 0]\[LL; 0];
            v = tmp(1:nx);
            c = tmp(end);
            
        elseif firstexit,
            v(~isgoal) = (eye(sum(~isgoal))-PP(~isgoal,~isgoal))\L(~isgoal);
            v(isgoal) = 0;
        end

        % compute Hamiltonian using policy-specific v
        for iu = 1:nu
            H(:,iu) = L(:,iu) + alpha*P(:,:,iu)*v;
        end

        % update policy and value
        [v, policy] = min(H,[],2);

        % check for convergence
        if max(abs(v-vold))<tol,
            break;
        end
    end


%--------- linear programming -------------------------------------------
elseif linearprogramming,

    % initialize linprog coefficients: 
    %  min  f'*v  s.t.  A*v<=b, Aeq*v=beq
    if discounted,
        f = -ones(nx,1);
        A = ones(nx*nu,nx);
        b = L(:);
        
    elseif average,
        f = [zeros(nx,1); -1];
        A = ones(nx*nu,nx+1);
        b = L(:);
        Aeq = [ones(1,nx), 0];
        beq = 0;
        
    elseif firstexit,
        nn = sum(~isgoal);
        f = -ones(nn,1);
        A = ones(nn*nu,nx);
        tmp = L(~isgoal,:); 
        b = tmp(:);
    end        
    
    % fill A matrix
    ic = 1;
    for iu = 1:nu
        for ix = 1:nx
            if ~(isgoal(ix) && firstexit),
                A(ic,1:nx) = -alpha*P(ix,:,iu);
                A(ic,ix) = 1-alpha*P(ix,ix,iu);
                ic = ic+1;
            end
        end
    end
    
    % run linprog
    if discounted,
        v = linprog(f,A,b);
    
    elseif average,
        tmp = linprog(f,A,b,Aeq,beq);
        v = tmp(1:nx);
        c = tmp(end);

    elseif firstexit,
        A = A(:,~isgoal);
        tmp = linprog(f,A,b);
        v(~isgoal) = tmp;
        v(isgoal) = 0;
    end
    
end


% %--------- plot value function and policy -------------------------------
% 
% figure(2);
% clf;
% sz = sqrt(nx);
% V = reshape(v,[sz,sz]);
% imagesc(V');
% axis image off;
% 
% if ~linearprogramming,
%     hold on;
%     rad = (sqrt(nu)-1)/2;
%     U1 = floor((policy-1)/(2*rad+1))-rad;
%     U2 = mod(policy-1,2*rad+1)-rad;
%     set(quiver(reshape(U1,[sz,sz])',reshape(U2,[sz,sz])','k'),'linewidth',1);
% end


%--------- compute address in transition matrix -------------------------

function i = adr(x1,x2,sz)

% wrap around
x1 = mod(x1-1,sz)+1;
x2 = mod(x2-1,sz)+1;

i = x1 + (x2-1)*sz;




%---------- construct the MDP -------------------------------------------

function [Pssa, L] = makeMDP

Pssa = zeros(2,2,2);
p=0.1;

Pssa(1,1,1) = 1-p;
Pssa(1,2,1) = p;
Pssa(1,1,2) = 0.1;
Pssa(1,2,2)=0.9;
Pssa(2,1,1)=0.3;
Pssa(2,2,1)=0.7;
Pssa(2,1,2)=0.8;
Pssa(2,2,2)=0.2;
L(1,1)=1;
L(2,1)=10;

