function [ X,X1,out] = transformPCA(X,X1,opts)
%transform the labeled and unlabeled data with PCA preserving preserve 
%variance preserved
DEF.preserve=75;
DEF.k=3;
if nargin <3
    opts=DEF;
else
    opts=getOptions(opts,DEF);
end
data=[X;X1];
ssX=size(X,1);
Z=zscore(data);
%ssX1=size(X1,1);
[pcs,tdata,varr,~,varr_explained,~]=pca(Z);
cum_ve=cumsum(varr_explained);
%k=find(cum_ve>=opts.preserve,1,'first');
X=tdata(1:ssX,1:opts.k);
X1=tdata(ssX+1:end,1:opts.k);
out.pcs=pcs;
out.cum_ve=cum_ve;
out.ve=cum_ve(3);
end

