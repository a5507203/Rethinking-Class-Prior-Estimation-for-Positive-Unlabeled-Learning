function [khat10,khat01,best_params] = mpeROC(sample1, sample0)
% MPEROC mixture proportion estimation based on the method of Blanchard et
% al. (2010) based on an ROC
%
% VARIABLES
%
% sample1: n1 x dim
% sample0: n0 x dim
%
% khat10 = kappahat(sample1|sample0): estimated mixture proportion (CPE method)
% khat01 = kappahat(sample0|sample1): estimated mixture proportion (CPE method)
% best_params: params selected by cross validation

% Code developed by Tyler Sanderson, Clayton Scott, and Daniel LeJeune
% University of Michigan 2013-2016
% 

[eta,y,nvec,best_params] = cpe(sample1,sample0,.2);

% nvec = [n0a n1a n0b n1b];
khat10 = kappahat(eta, y, nvec(4), nvec(3));
khat01 = kappahat(1-eta, -y, nvec(3), nvec(4));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Helper functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function s = kappahat(posteriors, Y, n1, n2)
%KAPPAHAT  Estimate maximum mixing proportion as min slope of ROC
%
% Given estimated class posteriors, 
% estimate maximum mixture proportion via method of 
% Blanchard et al. (2010) as detailed in the long version of
% Scott et al., "Classification with Asymmetric Label Noise"

% Extract the empirical false and true positive probabilities
nauc = 200; % number of sections when calculating AUC
[a,fp,tp] = auc(posteriors, Y, nauc); % calculate false pos/neg
ind = fp < 1 & fp > 0; % omit endpoints if present
false_positive_rates = fp(ind);
true_positive_rates = tp(ind);
n_roc = length(false_positive_rates);

% Estimate slopes using binomail tail inversion
numer = zeros(1,n_roc);
denom = zeros(1,n_roc);
delta = 0.1; % confidence level
for i=1:n_roc
    numer(i) = bininv(n2,n2*(1-true_positive_rates(i)),delta);
    denom(i)= 1 - bininv(n1,n1*false_positive_rates(i),delta);
end
% In theory, because of the union bound, delta should be divided by
% 2*n_roc, which makes the estimate a high probability upper bound on the
% true parameter. Not rescaling delta circumvents the looseness of the
% union bound, and also empirically decreases the bias of the estimate

% take minimum estimated slop
slopes = numer ./ denom;
s = min( slopes );

if 0 % show ROC and slope
    figure('units','normalized','outerposition',[0 0 1 1])
    subplot(121)
    plot(false_positive_rates, true_positive_rates);
    title('empirical ROC');
    hold on
    plot(1-denom, 1-numer, 'b:')
    subplot(122)
    plot((1 - true_positive_rates) ./ (1 - false_positive_rates))
    hold on
    plot(slopes, ':');
    title(num2str(s));
    pause
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [p] = bininv(m,e,c)
%BININV Binomal tail bound based on exact inversion of the tail
%
% m = sample size
% e = number of errors
% c = confidence parameter; probabality that bound is violated
%
% returned p estimates the exact p such that Pr(X \ge e) = c, where 
% X follows a binomial(m,p) distribution. 
%
% exact tail inversion is a tighter alternative to Chernoff's bound 

plo = 0;
phi = 1;
p = .5;

max_iter = 20;   % 
tol = c*0.001;    % tolerance 

% bisection search
iter = 0;
while iter <= max_iter
    iter = iter + 1;
    bintail = binocdf(e,m,p); % matlab command; relies on built in function gammaln
    if abs(bintail - c) <= tol
        return
    end
    if bintail < c
        phi = p;
    else
        plo = p;
    end    
    p = (phi + plo)/2;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [ area, alpha, beta ] = auc( eta, Y, res )
%AUC Compute the AUC and ROC of a classifier given the probability of a positive.
%
%   Input: 
%   eta - vector of probabilites of a positive
%   Y - vector of true labels
%   res - number of threshold points to test, defaults to 20
%
%   Output:
%   area - AUC
%   alpha - vector of probabilites of a false positive
%   beta - vector of probabilities of a true positive

if ~exist('res','var')
    res = 20;
end

n_pos = sum(Y == 1);
n_neg = sum(Y == -1);
alpha = zeros(res,1);
beta = zeros(res,1);
thresh = linspace(1,0,res);
for i=1:res
    t = thresh(i);
    alpha(i) = sum((eta > t).*(Y == -1))/n_neg;
    beta(i) = sum((eta > t).*(Y == 1))/n_pos;
end

area = trapz(alpha,beta);
