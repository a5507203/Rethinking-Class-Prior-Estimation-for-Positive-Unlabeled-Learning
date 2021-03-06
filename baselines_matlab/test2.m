addpath('./distributions/','./Algorithms/','./Transformations/','./shared/','./SVM/','./densityEstimation/','./plots/','./posterior/','./clustering/');

function [est, out] = apa()
n = 1000;     % number of negatives
n1 = 250;     % number of positives
alpha = 0.15; % true mixing proportion (class prior)

% the number of positive and negative data points in unlabeled data
p(2) = round(alpha * n)
p(1) = n - p(2);

% Means and covariance matrices of the two Gaussian distributions
m = [0, 10];           % true means
S = [1, 1];

%% Generate data sets X and X1
X = [];
for k = 1 : 2
    X = [X; repmat(m(k), p(k), 1) + randn(p(k), 1) * S(k)];
end

X1 = repmat(m(2), n1, 1) + randn(n1, 1) * S(2);

%% Get to estimation work
opts.postTransform=@cdfGaussTransform;
%opts.transform=@(X,X1)(transform_nn(X,X1,struct('h',7)));
%opts.transform=@(X,X1)(transform_nncv(X,X1,struct('h',7)))
%[est,out]= alphamaxB(X, X1, opts);
[est, out] = estimateMixprop(X, X1,'AlphaMax',opts);
