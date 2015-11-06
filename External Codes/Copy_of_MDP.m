% Illustration of value iteration, policy iteration,
% and linear programming approach to solving an MDP.
%
% Emanuel Todorov, UW 2010


function MDP

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
[P, L, isgoal] = makeMDP;
[nx, nu] = size(L);

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
            H(:,iu) = L(:,iu) + alpha*P(:,:,iu)*v;
        end

        % update v, compute policy
        [v, policy] = min(H,[],2);
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


%--------- plot value function and policy -------------------------------

figure(2);
clf;
sz = sqrt(nx);
V = reshape(v,[sz,sz]);
imagesc(V');
axis image off;

if ~linearprogramming,
    hold on;
    rad = (sqrt(nu)-1)/2;
    U1 = floor((policy-1)/(2*rad+1))-rad;
    U2 = mod(policy-1,2*rad+1)-rad;
    set(quiver(reshape(U1,[sz,sz])',reshape(U2,[sz,sz])','k'),'linewidth',1);
end


%--------- compute address in transition matrix -------------------------

function i = adr(x1,x2,sz)

% wrap around
x1 = mod(x1-1,sz)+1;
x2 = mod(x2-1,sz)+1;

i = x1 + (x2-1)*sz;




%---------- construct the MDP -------------------------------------------

function [P, L, isgoal] = makeMDP

% user-defined parameters 
sz = 20;                            % grid size
goal = [0.3, 0.6];                  % goal state (fraction of sz)
rad = 2;                            % neighborhood radius > 0
ppeak = 0.8;                        % prob. of going to selected state
ucost = 0.1;                        % weigth on quadratic control cost

% drift in the dynamics
drift = @(x) [-(x(2)-(sz+1)/2), abs(x(1)-(sz+1)/2)]*0.2;


nx = sz^2;                          % number of states
nu = (1+2*rad)^2;                   % number of actions per state

P = zeros(nx,nx,nu);                % transition probabilities
L = zeros(nx,nu);                   % costs
DRIFT = zeros(sz,sz,2);             % drift evaluated at each state

% loop over states
for x1 = 1:sz
    for x2 = 1:sz
        next = round([x1, x2] + drift([x1, x2]));
        DRIFT(x1,x2,:) = round(drift([x1, x2]));

        % loop over controls
        u = 1;
        for u1 = -rad:rad
            for u2 = -rad:rad
                P(adr(x1,x2,sz), adr(next(1)+u1,next(2)+u2,sz), :) = ...
                    (1-ppeak)/(nu-1);

                P(adr(x1,x2,sz), adr(next(1)+u1,next(2)+u2,sz), u) = ...
                    ppeak;

                L(adr(x1,x2,sz), u) = 1 + ucost*(u1^2 + u2^2);

                u = u+1;
            end
        end
    end
end

% goal state
isgoal = (1:nx)'==adr(round(goal(1)*(sz-1))+1,round(goal(2)*(sz-1))+1,sz);
L(isgoal,:) = L(isgoal,:)-1;


% plot dynamics and state cost
figure(1);
clf;
Q = ones(sz,sz);
Q(isgoal) = 0;
imagesc(Q');
hold on;
set(quiver(DRIFT(:,:,2),DRIFT(:,:,1),'k'),'linewidth',1);
axis image off;
