function khat10 = roc(a,b)

X = str2num(a);
X1 = str2num(b);



[khat10,khat01,best_params] = mpeROC(X, X1);


end