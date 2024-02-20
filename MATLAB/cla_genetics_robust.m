% Given a particular lambda value, this code find the point on the
% critical frontier for that lambda.

% set up the problem manually using variables
n  = 3;
S  = [1];  % MATLAB will make a double, but that's fine
D  = [2,3];
mu = [1; 5; 2];
lb = [0.0; 0.0; 0.0];
ub = [1.0; 1.0; 1.0];
covar = [
    1, 0, 0;
    0, 5, 0;
    0, 0, 3;
];

% robust optimization variables
omega = diag(rand(n,1));  % TODO work out how best to set this param
kappa = rand;             % TODO work out how best to set this params

% lagrange multiplier
lambda = 0;

% define problem variable; incorporates absolutethe upper and lower bounds
portprob = optimproblem;
w = optimvar('w',n);

% define problem objective function
objective = w'*covar*w/2  - lambda*w'*mu - kappa*(w'*omega*w)^0.5;
portprob.Objective = objective;

% define problem constraints
portprob.Constraints.siresum = w(S)'*ones(length(S),1) == 1/2;
portprob.Constraints.damsum  = w(D)'*ones(length(D),1) == 1/2;
portprob.Constraints.lowerbound = w >= lb;   % weights lower bound
portprob.Constraints.upperbound = w <= ub;   % weights upper bound

% toolbox options
options = optimoptions('fmincon','Display','iter','TolFun',1e-10);

% initial point
w0 = struct('w',[0.5;0.5;0.0]);

tic
[w_opt, objective_value_opt] = solve(portprob,w0,'Options',options);
toc

w_opt.w