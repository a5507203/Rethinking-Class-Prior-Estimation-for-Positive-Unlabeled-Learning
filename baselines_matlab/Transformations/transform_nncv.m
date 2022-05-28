function [x,x1,out] = transform_nncv(X0,X1,opts)
%Transforms multivariate data into univariate data.  Uses neural networks 
%to estimate posterior.
%% All parameters are hard-coded here
DEF.B=100;% 100 bagged neural networks
DEF.h=5;  % 5 hidden neurons
DEF.l=2; %2 layers
DEF.val_frac=0.25; % fraction of training data in the validation set
DEF.K=10;
if nargin < 3
    opts=DEF;
else
    opts=getOptions(opts,DEF);
end

K=opts.K;
B = opts.B;
h = opts.h;  
l=opts.l;
val_frac = opts.val_frac; 
% X = training data, where rows are data points and columns are features
X=[X0;X1];
[~, ~, X] = normalize(X, [], []);
% s = class labels, where 0 means negatives and 1 means positives
s=[zeros(size(X0,1),1);ones(size(X1,1),1)];
% predictions (cummulative)
pX = zeros(size(X, 1), 1);


indices = crossvalind('Kfold',size(X,1),10);
%Initialize an object to measure the performance of the classifier.
%cp = classperf(D);
%Perform the classification using the measurement data and report the error rate, which is the ratio of the number of incorrectly classified samples divided by the total number of classified samples.
ix=1:size(X,1);
for kk = 1:K
    val_ind = (indices == kk); 
    tra_ind = ~val_ind;
    
    net{kk} = feedforwardnet([h], 'trainrp');
    for jj=1:l
        net{kk}.layers{jj}.transferFcn = 'tansig';   
    end
    %net{b}.layers{1}.transferFcn = 'tansig';
    %net{b}.layers{2}.transferFcn = 'tansig';
    net{kk}.trainParam.epochs = 500;
    net{kk}.trainParam.show = NaN;
    net{kk}.trainParam.showWindow = false;
    net{kk}.trainParam.max_fail = 25;
    net{kk}.divideFcn = 'divideind';
    %net{b}.divideParam.trainInd = 1 : size(Xb, 1);
    %net{b}.divideParam.valInd = [];
    net{kk}.divideParam.trainInd = ix(tra_ind);
    net{kk}.divideParam.valInd = ix(val_ind);
    net{kk}.divideParam.testInd = [];

    net{kk} = train(net{kk},X', s');
    
    % apply the neural network to the out-of-bag data
    p = sim(net{kk}, X');
    pX= pX + p';
end
pX=pX/K;
g=pX;

auc = get_auc_ultra(g, s);

x_ind=1:size(X0,1);
x=g(x_ind)';
x1=g(setdiff(1:size(X,1),x_ind))';

out.opts=opts;
out.x=x;
out.x1=x1;
out.auc=auc;
end

