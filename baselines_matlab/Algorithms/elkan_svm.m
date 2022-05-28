function [alpha, auc,g, out] = elkan_svm (X, s,opts)

% X = training data, where rows are data points and columns are features
% s = class labels, where 0 means unlabeled and 1 means labeled

%% All parameters are hard-coded here
DEF.N = 10; % Cross-validation parameter

% learning and SVMlight parameters
DEF.info.do_normalize = 1; % z-score normalization if 1, nothing if 0
DEF.info.pos_weight = 1;   % equally weight positives and negatives (balanced)
DEF.info.kernel = 1;       % 1 = polynomial kernel, 2 = RBF kernel
DEF.info.parameter = 2;    % if kernel = 1, this is poly degree
% if kernel = 2, this is sigma

%Change the path to the location of SVM Light on your system
if ismac()
    DEF.info.SVMlightpath = '';
elseif isunix()
    DEF.info.SVMlightpath = '~/software/svm_light';
elseif ispc()
    DEF.info.SVMlightpath = '';
else
    error('My SVM directory cannot be set');
end
if nargin < 3
    opts=DEF;
else
    if isfield(opts,'info')
        opts.info=getOptions(opts.info,DEF.info);
    end
    opts=getOptions(opts,DEF);
end
info=opts.info;
N=opts.N;
%% Estimation process

% predictions (cummulative)
pX = zeros(size(X, 1), 1);

% number of predictions for each data point
npX = zeros(size(X, 1), 1);

% 10-fold cross-validation
b = n_fold(size(X, 1), N);

% run training and testing
for i = 1 : N
    q = setdiff(1 : size(X, 1), b{i});
    Xtr = X(q, :);
    ytr = s(q, :);
    Xts = X(b{i}, :);

    % normalize training and test sets
    if info.do_normalize == 1
        [mn, sd, Xtr] = normalize(Xtr, [], []);
        [~, ~, Xts] = normalize(Xts, mn, sd);
    end

    p = SVMprediction([Xtr ytr], Xts, info);
    
    pX(b{i}) = p;  % add predictions
end

% Platt's correction to get posterior probabilities
w = weighted_logreg(pX, s, ones(size(s, 1), 1));
g = 1 ./ (1 + exp(-w(1) - w(2) * pX));

% check accuracy of the predictor
auc = get_auc_ultra(g, s);

% estimate alpha based on the type of estimator from the paper

q0 = find(s == 0);

% estimator e1 for c
c = sum(g(s == 1)) / length(find(s == 1));
w = (1 - c) / c * (g(q0) ./ (1 - g(q0)));
alpha(1) = length(find(s == 1)) / length(s) / c;        % alternative e1
alpha(4) = (length(find(s == 1)) + sum(w)) / length(s); % main e1

% estimator e2 for c
c = sum(g(s == 1)) / sum(g);
w = (1 - c) / c * (g(q0) ./ (1 - g(q0)));
alpha(2) = length(find(s == 1)) / length(s) / c;         % alternative e2
alpha(5) = (length(find(s == 1)) + sum(w)) / length(s);  % main e2

% estimator e3 for c
c = max(g);
w = (1 - c) / c * (g(q0) ./ (1 - g(q0)));
alpha(3) = length(find(s == 1)) / length(s) / c;         % alternative e3
alpha(6) = (length(find(s == 1)) + sum(w)) / length(s);  % main e3


out.opts=opts;
out.alpha=alpha;
out.auc=auc;
out.g=g;
return
