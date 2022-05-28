function [eta,labels,nvec,best_params] = cpe(sample1, sample0,train_frac)
% CPE class probability estimation using kernel logistic regression
%
% Important note: Users will need to set the appropriate path for Liblinear
% executables below
%
% VARIABLES
%
% sample1: n1 x dim
% sample0: n0 x dim
% train_frac: fraction of data on which to train kernel logistic
%       regression; the rest is held out and the fitted model is evaluated 
%       on the held out points to obtain estimates of the class probability
%
% eta: vector of estimated class probabilities for held out data
% labels: vector of labels associated to data points whose class
%       probabilities are estimated by eta
% nvec: vector of samples counts by class in both portions of data
% best_params: params selected by maximizing a cross validation estimate of
%       the AUC (kernel bandwidth, KLR regularization parameter)
%
% Note: The second argument is called sample1 here because it makes sense
% to think of it as the "negative class" in the classification methods used
% below

% Code developed by Tyler Sanderson, Clayton Scott, and Daniel LeJeune
% University of Michigan 2013-2016
% 

p = size(sample1,2); % dimension
n1 = size(sample1,1); 
n0 = size(sample0,1);
n = n1+n0;
X = [sample0; sample1];
Y = [-ones(n0,1); ones(n1,1)]; % logistic regression routine below (from liblinear) 
                                % uses +1/-1 label convention
perm = randperm(n);
X = X(perm,:);
Y = Y(perm,:);

% set up data for hold out estimation
a_idx = 1:floor(train_frac*n);
b_idx = setdiff(1:n,a_idx); % estimate class probabilities on this portion
Xa = X(a_idx,:);
Xb = X(b_idx,:);
na = size(Xa,1);
nb = size(Xb,1);
Ya = Y(a_idx);
n0a = sum(Ya == -1);
n1a = sum(Ya == 1);
Yb = Y(b_idx);
n0b = sum(Yb == -1);
n1b = sum(Yb == 1);
nvec = [n0a n1a n0b n1b];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Apply kernel logistic regression to get empirical ROC. KLR is implemented 
% with a combination of random fourier features (Rahimi and Recht 2007) 
% and liblinear (http://www.csie.ntu.edu.tw/~cjlin/liblinear)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% jaakkola heuristic for Gaussian kernel bandwidth
dist_mat = sqrt(dist2(Xa(find(Ya == -1),:),Xa(find(Ya == 1),:)));
jaakk_heur = median(dist_mat(:));

% modify this line for your system
path('/home/yyao0814/Documents/liblinear/matlab',path) % add liblinear executables "train" and "predict"

% sample random Fourier directions and angles
D = 500; % RFF dimension
Omega = randn(p,D); % RVs defining RFF transform
beta = rand(1,D)*2*pi; 
nauc = 20; % number of sections when calculating AUC

% Select parameters using cross-validation, maximizing AUC
best_auc = 0;
for sigma=jaakk_heur*2.^linspace(-3,1,5) % kernel bandwidth
    Z = transformFeatures(Xa/sigma,Omega,beta); % calculate RFFs
    for lambda=10.^linspace(-2,2,5); % regularization parameter
        cv_idx = crossvalind('Kfold',na,5);
        cv_auc = zeros(5,1);
        for i=1:5
            Y_train = Ya(cv_idx ~= i);
            Z_train = Z(cv_idx ~= i,:);
            Y_test = Ya(cv_idx == i);
            Z_test = Z(cv_idx == i,:);
            model = train(Y_train,sparse(Z_train),['-q -s 0 -c ', ...
                num2str(1/lambda)]);
            [~,~,eta] = predict(Y_test,sparse(Z_test),model,'-q -b 1');
            cv_auc(i) = auc(eta(:,1),Y_test);
        end
        mean_auc = mean(cv_auc);
        %fprintf('lambda=%.3f,sigma=%.3f,AUC=%.3f\n', lambda, sigma, mean_auc)
        if mean_auc > best_auc
            best_auc = mean_auc;
            best_params = [lambda, sigma];
        end
    end
end

% train on the whole set
lambda = best_params(1);
sigma = best_params(2);
Za = transformFeatures(Xa/sigma,Omega,beta);
model = train(Ya,sparse(Za),['-q -s 0 -c ', num2str(1/lambda)]);

% calculate predictions for held out data
Zb = transformFeatures(Xb/sigma,Omega,beta);
[Y_tilde,~,pred] = predict(Yb,sparse(Zb),model,'-q -b 1');
eta = pred(:,1);
labels = Yb;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [ Z ] = transformFeatures( X, Omega, beta )
%TRANSFORMFEATURES Transforms data to the random Fourier feature space
%
%   Input: 
%   X - n x p data matrix (each row is a sample) 
%   Omega - p x D matrix of random Fourier directions (one for each
%   dimension of a sample x)
%   beta - 1 x D vector of random angles
%
%   Output:
%   Z - n x D matrix of random Fourier features

D = size(Omega,2);
Z = cos(bsxfun(@plus,X*Omega,beta))*sqrt(2/D);

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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function n2 = dist2(x, c)
%DIST2	Calculates squared distance between two sets of points.
%
%	Description
%	D = DIST2(X, C) takes two matrices of vectors and calculates the
%	squared Euclidean distance between them.  Both matrices must be of
%	the same column dimension.  If X has M rows and N columns, and C has
%	L rows and N columns, then the result has M rows and L columns.  The
%	I, Jth entry is the squared distance from the Ith row of X to the
%	Jth row of C.
%
%	See also
%	GMMACTIV, KMEANS, RBFFWD
%

%	Copyright (c) Christopher M Bishop, Ian T Nabney (1996, 1997)
%	Modified for speedup using bsxfun() by SJ Yoon, Univ. of Michigan (2014)

[~, dimx] = size(x);
[~, dimc] = size(c);
if dimx ~= dimc
	error('dist2.m: Data dimension does not match dimension of centres')
end

n2 = bsxfun(@plus, sum((x.^2),2), sum((c.^2),2)') - 2*(x*(c'));

