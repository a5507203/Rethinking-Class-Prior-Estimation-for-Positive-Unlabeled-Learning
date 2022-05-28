function [alpha, gamma, pnPost, pnPost1, out] = runPreClustBias(S) 

X=S.X;
X1=S.X1;
opts=struct();
opts.C1=S.cluster_X1;
opts.C=S.cluster_X;
[alpha, gamma, eta,  alphaC, clProp, clProp1, out]=alphamaxB(X,X1,opts);
[pnPost,pnPost1]=recoverPNPostC(alpha,gamma,eta,alphaC,out);

end
