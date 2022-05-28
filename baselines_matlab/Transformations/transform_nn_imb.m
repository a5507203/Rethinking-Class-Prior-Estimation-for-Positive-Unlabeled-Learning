function [x,x1,out] = transform_nn_imb(X0,X1,opts)
%Transforms multivariate data into univariate data.  Uses neural networks 
%to estimate posterior. bootstrapping is done separately from the positives
%and negatives
%% All parameters are hard-coded here
DEF.B=100;% 100 bagged neural networks
DEF.h=5;  % 5 hidden neurons
DEF.l=2; %2 layers
DEF.val_frac=0.25; % fraction of training data in the validation set
DEF.Xe=[];
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
[~, ~, X] = normalize(X, [], []);
n0=size(X0,1);
n1=size(X1,1);
n=size(X,1);
% s = class labels, where 0 means negatives and 1 means positives
s=[zeros(n0,1);ones(n1,1)];
% predictions (cummulative)
pX = zeros(n, 1);

% number of predictions for each data point
npX = zeros(n, 1);
pXe=zeros(size(opts.Xe,1),1);
%npXe=zeros(size(Xe,1),1);

% run training and testing
for b = 1 : B
    %b 
    q0 = unique(ceil(rand(1, n0) * n0));
    q0_c = setdiff(1 : n0, q0); % remaining data points
    q1 = unique(ceil(rand(1, n1) * n1));
    q1_c = setdiff(1 : n1, q1); % remaining data points
    q1 = q1+n0;
    q1_c = q1_c+n0;
    q0val = q0(randperm(length(q0),ceil(val_frac*length(q0))));
    q1val = q1(randperm(length(q1),ceil(val_frac*length(q1))));
    q0train = setdiff(q0, q0val); 
    q1train = setdiff(q1, q1val); 
    q1val=duplicate(q1val,length(q0val));
    q1train=duplicate(q1train,length(q0train));
    q0val=duplicate(q0val,length(q1val));
    q0train=duplicate(q0train,length(q1train));
    q=[q0train,q1train,q0val,q1val];
    
    q_c=[q0_c, q1_c];
    Xb = X(q, :);
    yb = s(q, :);
    Xt = X(q_c,:);
    nb=size(Xb,1);
    %val_ind=[q0val,q1_val];
    tra_ind=1:(length(q0train)+length(q1train));
    val_ind=(length(tra_ind)+1):nb;
    % sample with replacement from the data
    %q = ceil(rand(1, size(X, 1)) * size(X, 1));
    %q_c = setdiff(1 : size(X, 1), q); % remaining data points
    %Xb = X(q, :);

    % set class labels for training
    
    % indices for the validation set
    %q0val = q0(randperm(length(q0)));
    %q1val = randperm(length(q1))+length(q0);
    %val_ind0 = q0val(1 : floor(val_frac * length(q0val)));
    %val_ind1 = q1val(1 : floor(val_frac * length(q1val)));
    %val_ind=[val_ind0,val_ind1];
    %tra_ind = setdiff(1 : nb, val_ind);
    
    % normalize training and test sets
    %[mn, sd, Xb] = normalize(Xb, [], []);
    %[~, ~, Xt] = normalize(X(q_c, :), mn, sd);
    
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
    if (length(opts.Xe)>0)
        pe = sim(net{b}, opts.Xe');
        pXe = pXe + pe';  % add predictions
    end
end

% this will produce values for function g, using out-of-bag data
q = find(npX ~= 0); % just in case some data points haven't been selected
g(q) = pX(q) ./ npX(q);
if length(q) < size(X, 1)
    g(setdiff(1 : size(X, 1), q)) = mean(g(q));
end
if (length(opts.Xe)>0)
      pXe = pXe/B;  % add predictions
end
%g=PNpostCal_imb(g,0.5,n1/n);
auc = get_auc_ultra(g, s);

x_ind=1:size(X0,1);
x=g(x_ind)';
x1=g(setdiff(1:size(X,1),x_ind))';

out.opts=opts;
out.x=x;
out.x1=x1;
out.auc=auc;
out.pp=0.5;
out.poste=pXe;

    function v = duplicate(x,size)
        v=x;
        size_dif = size-length(x);
        if size_dif>0
            ix=randi(length(x),size_dif,1);
            v((length(x)+1):(length(x)+size_dif))=x(ix);
        end
    end
end

