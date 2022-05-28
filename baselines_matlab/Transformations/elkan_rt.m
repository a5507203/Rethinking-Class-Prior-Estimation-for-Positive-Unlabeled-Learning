function [alpha, auc, g] = elkan_rt (X, s)

% X = training data, where rows are data points and columns are features
% s = class labels, where 0 means unlabeled and 1 means labeled

%% All parameters are hard-coded here
B = 250; % 250 bagged regression trees

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

    t{b} = treefit(Xb, yb, 'splitmin', 1);
    
    % apply the tree to the out-of-bag data
    p = treeval(t{b}, X(q_c, :));

    pX(q_c) = pX(q_c) + p;  % add predictions
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

% find unlabeled and labeled examples
q0 = find(s == 0);
q1 = find(s == 1);

% estimator e1 for c
c = sum(g(s == 1)) / length(q1);
w = (1 - c) / c * (g(q0) ./ (1 - g(q0)));
alpha(1) = length(q1) / length(s) / c;
alpha(4) = (length(q1) + sum(w)) / length(s);

% estimator e2 for c
c = sum(g(s == 1)) / sum(g);
w = (1 - c) / c * (g(q0) ./ (1 - g(q0)));
alpha(2) = length(q1) / length(s) / c;
alpha(5) = (length(q1) + sum(w)) / length(s);

% estimator e3 for c
c = max(g);
w = (1 - c) / c * (g(q0) ./ (1 - g(q0)));
alpha(3) = length(q1) / length(s) / c;
alpha(6) = (length(q1) + sum(w)) / length(s);

return
