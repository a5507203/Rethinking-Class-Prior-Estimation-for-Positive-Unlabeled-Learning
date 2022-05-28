function [alpha, gamma, pnPost, pnPost1, out] = runBias(S) 

X=S.X;
X1=S.X1;
opts=struct();
opts.M=5;
[alpha, gamma, eta, alphaC, clProp, clProp1, out]=alphamaxB(X,X1,opts);
[pnPost,pnPost1]=recoverPNPostC(alpha,gamma,eta,alphaC,out);

end
