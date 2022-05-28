addpath('./distributions/','./Algorithms/','./Transformations/','./shared/','./SVM/','./densityEstimation/','./plots/','./posterior/','./clustering/');

S=load('~/Data/Bias/finalData.mat');
alpha=S.alpha;
opts=struct();
opts.Xe=S.X1;
%global post
%global post1
[post,p1,out]=transform_nn_imb(S.X0_N,S.X0_P,opts);
post1=out.poste;
post=changePrior([p1;post],out.pp,alpha);
post1=changePrior(post1,out.pp,alpha);

[alpha_1,gamma_1, post_1,post1_1,out_1]=runPreClustBias(S);
[alpha_2,gamma_2, post_2, post1_2, out_2]=runBias(S);
[alpha_3, post_3, post1_3, out_3]=runUnBias(S);

%x=post;
%y=post_1;
%scatter(x,y,'r.')
%hold on
%[n,c] = hist3([x, y]);
%contour(c{1},c{2},n)
%[x,x1,outtt]=transform_nn_imb(S.X(C==3,:),S.X1(C==3,:),opts);

