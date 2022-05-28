clear
clc

n = 1000;     % number of negatives
n1 = 250;     % number of positives
alpha = 0.75; % true mixing proportion (class prior)

% the number of positive and negative data points in unlabeled data
p(2) = round(alpha * n)
p(1) = n - p(2);

% Means and covariance matrices of the two Gaussian distributions
m = [[4 4]' [6 3]'];           % true means
S(:, :, 1) = [1 0.75; 0.75 1]; % true covariance matrices
S(:, :, 2) = [1 0.75; 0.75 1]; 

%m = [[4 4]' [6 3]'];             % true means
%S(:, :, 1) = [1 -0.75; -0.75 1]; % true covariance matrices
%S(:, :, 2) = [1 0.75; 0.75 1]; 

%m = [[1 2]' [6 3]'];           % true means
%S(:, :, 1) = [1 0; 0 1];       % true covariance matrices
%S(:, :, 2) = [1 0.75; 0.75 1]; 

%% Generate data sets X and X1
X = [];
for k = 1 : 2
    X = [X; repmat(m(:, k)', p(k), 1) + randn(p(k), 2) * chol(S(:, :, k))];
end

X1 = repmat(m(:, 2)', n1, 1) + randn(n1, 2) * chol(S(:, :, 2));

opts=struct();
opts.h=1;
[x,x1,out]=transform_nn(X,X1,opts);
pp=length(x1)/(length(x1)+length(x));
xpn=PUpost2PNpost(x,(1-pp)/pp,0.15);
[y,y1,out]=transform_nn_imb(X,X1,opts);
ypn=PUpost2PNpost(y,1,0.15);


opts.Xe=X1;
[z,z1,out]=transform_nn_imb(X(1:p(1),:),X((p(1)+1):end,:),opts);
%pp=length(x1)/(length(x1)+length(x));
zpn=changePrior([z;z1], 0.5, 0.15);
figure;
scatter(zpn,xpn)
xlabel('posterior PN')
ylabel('corrected PN posterior, imbalance training data')
figure;
scatter(zpn,ypn)
xlabel('posterior PN')
ylabel('corrected PN posterior, rebalanced training data')

%% Get to estimation work
%opts.postTransform=@cdfGaussTransform;
opts.transform=@(X,X1)(transform_nn(X,X1,struct('h',7)));
%opts.transform=@(X,X1)(transform_nncv(X,X1,struct('h',7)))
%[est,out]= alphamaxB(X, X1, opts);
[est, out] = estimateMixprop(X, X1,'AlphaMax',opts);
