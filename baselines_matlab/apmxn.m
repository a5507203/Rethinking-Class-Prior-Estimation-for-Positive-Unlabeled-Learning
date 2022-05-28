function est = apmxn(a,b)

X = str2num(a);
X1 = str2num(b);


%% Get to estimation work
%opts.postTransform=@cdfGaussTransform;
opts.transform=@(X,X1)(transform_nn(X,X1,struct('h',7)));
%opts.transform=@(X,X1)(transform_nncv(X,X1,struct('h',7)))
%[est,out]= alphamaxB(X, X1, opts);
[est, out] = estimateMixprop(X, X1,'AlphaMax_N',opts);
