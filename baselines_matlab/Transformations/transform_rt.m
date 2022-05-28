function [x,x1,out] = transform_rt (X0,X1,opts)

%Transforms multivariate data into univariate data.  Uses random forest 
%to estimate posterior.

%% All parameters are hard-coded here
DEF.B=250; % 250 bagged regression trees
if nargin < 3
    opts=DEF;
else
    opts=getOptions(opts,DEF);
end

B = opts.B; 

%% Estimation process

% X = training data, where rows are data points and columns are features
X=[X0;X1];
% s = class labels, where 0 means negatives and 1 means positives
s=[zeros(size(X0,1),1);ones(size(X1,1),1)];

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
auc = get_auc_ultra(g, s);
%% return with correct output
x_ind=1:size(X0,1);
x=g(x_ind)';
x1=g(setdiff(1:size(X,1),x_ind))';

out.auc=auc;
out.opts=opts;
out.x=x;
out.x1=x1;
out.auc;
return
