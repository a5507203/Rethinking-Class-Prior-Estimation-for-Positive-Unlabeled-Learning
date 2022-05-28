function [x,x1,out] = transform_svm(X0,X1,opts)
%Transforms multivariate data into univariate data.  Uses SVM 
%to estimate posterior.
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
%% Transform Data
% X = training data, where rows are data points and columns are features
X=[X0;X1];
% s = class labels, where 0 means negativea and 1 means positives
s=[zeros(size(X0,1),1);ones(size(X1,1),1)];
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

auc = get_auc_ultra(g, s);

x_ind=1:size(X0,1);
x=g(x_ind)';
x1=g(setdiff(1:size(X,1),x_ind))';

out.opts=opts;
out.x=x;
out.x1=x1;
out.auc=auc;
end



