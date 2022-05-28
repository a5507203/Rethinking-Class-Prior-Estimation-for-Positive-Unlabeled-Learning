function [x,x1,out] = transform_nn(X0,X1,opts)
%Transforms multivariate data into univariate data.  Uses neural networks 
%to estimate posterior.
%% All parameters are hard-coded here
DEF.B=100;% 100 bagged neural networks
DEF.h=5;  % 5 hidden neurons
DEF.l=2; %2 layers
DEF.val_frac=0.25; % fraction of training data in the validation set
if nargin < 3
    opts=DEF;
else
    opts=getOptions(opts,DEF);
end

B = opts.B; 
h = opts.h;  
l=opts.l;
val_frac = opts.val_frac; 
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

    % indices for the validation set
    qval = randperm(size(Xb, 1));
    val_ind = qval(1 : floor(val_frac * length(qval)));
    tra_ind = setdiff(1 : length(qval), val_ind);
    
    % normalize training and test sets
    [mn, sd, Xb] = normalize(Xb, [], []);
    [~, ~, Xt] = normalize(X(q_c, :), mn, sd);
    
    % initialize and train b-th neural network
    %net{b} = newff(Xb', yb', h, {'tansig', 'tansig'}, 'trainrp');
    net{b} = feedforwardnet([h], 'trainrp');
    for jj=1:l
        net{b}.layers{jj}.transferFcn = 'tansig';   
    end
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
%g=postCal(g);
auc = get_auc_ultra(g, s);

x_ind=1:size(X0,1);
x=g(x_ind)';
x1=g(setdiff(1:size(X,1),x_ind))';


out.opts=opts;
out.x=x;
out.x1=x1;
out.pp=length(x1)/(length(x1)+length(x));
out.auc=auc;
end

