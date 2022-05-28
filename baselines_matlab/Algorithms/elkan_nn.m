function [alpha, auc, g,out] = elkan_nn (X, s,opts)

% X = training data, where rows are data points and columns are features
% s = class labels, where 0 means unlabeled and 1 means labeled

%% All parameters are hard-coded here
DEF.B=100;% 100 bagged neural networks
DEF.h=5;  % 5 hidden neurons
DEF.val_frac=0.25; % fraction of training data in the validation set
if nargin < 3
    opts=DEF;
else
    opts=getOptions(opts,DEF);
end

B = opts.B; 
h = opts.h;  
val_frac = opts.val_frac; 
%% Estimation process

% predictions (cummulative)
pX = zeros(size(X, 1), 1);

% number of predictions for each data point
npX = zeros(size(X, 1), 1);

% run training and testing
for b = 1 : B
    %b 
    
    % sample with replacement from the data
    q = ceil(rand(1, size(X, 1)) * size(X, 1));
    q_c = setdiff(1 : size(X, 1), q); % remaining data points
    Xb = X(q, :);

    % set class labels for training
    yb = s(q, :);

    % indices for the validation set
    qval = randperm(size(Xb, 1));
    val_ind = qval(1 : floor(val_frac * length(qval)));
    tra_ind = setdiff(1 : length(qval), val_ind);
    
    % normalize training and test sets
    [mn, sd, Xb] = normalize(Xb, [], []);
    [~, ~, Xt] = normalize(X(q_c, :), mn, sd);
    
    % initialize and train b-th neural network
    net{b} = newff(Xb', yb', h, {'tansig', 'tansig'}, 'trainrp');
    %net{b} = feedforwardnet([h], 'trainrp');
    %net{b}.layers{1}.transferFcn = 'tansig';
    %net{b}.layers{2}.transferFcn = 'tansig';
    
    net{b}.trainParam.epochs = 500;
    net{b}.trainParam.show = NaN;
    net{b}.trainParam.showWindow = false;
    net{b}.trainParam.max_fail = 25;
    net{b}.divideFcn = 'divideind';
    %net{b}.divideParam.trainInd = 1 : size(Xb, 1);
    %net{b}.divideParam.valInd = [];
    net{b}.divideParam.trainInd = tra_ind;
    net{b}.divideParam.valInd = val_ind;
    net{b}.divideParam.testInd = [];

    net{b} = train(net{b}, Xb', yb');
    
    % apply the neural network to the out-of-bag data
    p = sim(net{b}, Xt');
    pX(q_c) = pX(q_c) + p';  % add predictions
    npX(q_c) = npX(q_c) + 1; % update counts
end

% this will produce values for function g, using out-of-bag data
q = find(npX ~= 0); % just in case some data points haven't been selected
g(q) = pX(q) ./ npX(q);
if length(q) < size(X, 1)
    g(setdiff(1 : size(X, 1), q)) = mean(g(q));
end

% check accuracy of the predictor
auc = get_auc_ultra(g, s);

% estimate alpha based on the type of estimator from the paper

% estimator e1 for c
%alpha(1) = length(find(s == 1)) ^ 2 / length(s) / sum(g(s == 1));
%  estimator e2 for c
%alpha(2) = length(find(s == 1)) / (length(s) * sum(g(s == 1)) / sum(g));
%  estimator e3 for c
%alpha(3) = length(find(s == 1)) / (length(s) * max(g));

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
