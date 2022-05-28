function [alpha, gamma, eta, alphaC, clProp, clProp1, out] = alphamaxB(X,X1,opts)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
addpath('./distributions/','./Algorithms/','./Transformations/','./shared/','./SVM/','./densityEstimation/','./plots/');
DEF.transform=@transform_nn_imb;
if nargin < 3
    opts=DEF;
else
   % opts.is_user_constraint = isfield(opts,'constraints');
    opts=getOptions(opts,DEF);
end
if isfield(opts,'C1')
    C1=opts.C1;
    C=opts.C;
else
    [C,C1,~,out.clust]=findClusters(X,X1,opts.M);
end
out.C=C;
out.C1=C1;
[mix_ss,mix_dim]=size(X);
[comp_ss,comp_dim]=size(X1);
n_clust=max(C1);
if mix_dim ~= comp_dim
    error('The dimensions of X and X1 are not equal')
end
Xs={};
X1s={};
xs={};
x1s={};
ams={};
ests=[];
cSize=[];
cSize1=[];
transforms={};
for i=1:n_clust
    Xs=[Xs,{X(C==i,1:end)}];
    X1s=[X1s,{X1(C1==i,1:end)}];
    cSize=[cSize,sum(C==i)]
    cSize1=[cSize1,sum(C1==i)]
    if (cSize(i)~=0)
        [x,x1,tr]=opts.transform(Xs{i},X1s{i});
        x=x(:); x1=x1(:);
        %[x,x1]=cdfGaussTransform(x,x1);
        xs=[xs,x]; x1s=[x1s,x1];transforms=[transforms,tr];
        [est,outAm]=estimateMixprop(x,x1,'AlphaMax');
    else
        xs=[xs,{{}}]; x1s=[x1s,ones(cSize1(i),1)];transforms=[transforms,{{}}];
        est=struct();
        est.AlphaMax=0;
        outAm=struct();
    end
    ests=[ests,est.AlphaMax];
    ams=[ams,outAm];
end
%Xclub=cat(1,xs{:});
%X1club=cat(1,x1s{:});
%[est,outAm]=estimateMixprop(Xclub,X1club,'AlphaMax');
%est=est.AlphaMax;

out.xs=xs; out.x1s=x1s;
out.ests=ests;
out.ams=ams;
%out.est=est;
out.cSize=cSize;
out.cSize1=cSize1;
out.transforms=transforms;
%gamma=ests.*cSize/sum(ests.*cSize);
gamma=ests.*cSize;
gamma=gamma/sum(gamma);
eta=(1-ests).*cSize;
eta= eta/sum(eta);

%alpha=[est,sum(ests.*cSize)/sum(cSize)];
alpha=sum(ests.*cSize)/sum(cSize);
alphaC=ests;
clProp=cSize/sum(cSize);
clProp1=cSize1/sum(cSize1);
end

