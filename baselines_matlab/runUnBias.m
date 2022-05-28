function [alpha, pnPost, pnPost1, out] = runUnBias(S) 

X=S.X;
X1=S.X1;
opts.transform=@transform_nn_imb;
[alpha, out]=estimateMixprop(X,X1,'AlphaMax',opts);
[pnPost,pnPost1]=recoverPNPost(alpha.AlphaMax,out);

end
