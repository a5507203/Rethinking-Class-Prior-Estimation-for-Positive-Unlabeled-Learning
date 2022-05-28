function [khat10,khat01,best_params,trueetatmaxprct,trueetatminprct] = mpeCPE(sample1, sample0, pi1t, pi0t)
% MPEROC mixture proportion estimation based on the method of Blanchard et
% al. (2010) based on an ROC
%
% VARIABLES
%
% sample1: n1 x dim
% sample0: n0 x dim
%
% pi1t: true pi1 tilde, for testing purposes (optional)
% pi0t: true pi0 tilde, for testing purposes (optional)
%
% khat10 = kappahat(sample1|sample0): estimated mixture proportion (CPE method)
% khat01 = kappahat(sample0|sample1): estimated mixture proportion (CPE method)
% best_params: params selected by cross validation

% Code developed by Tyler Sanderson, Clayton Scott, and Daniel LeJeune
% University of Michigan 2013-2016
% 

[etat,yt,nvec,best_params] = cpe(sample1,sample0,.80);

% nvec = [n0a n1a n0b n1b];
etamax = prctile(etat,95);
etamin = prctile(etat,5);
khat10 = (1/etamax - 1)*nvec(4)/nvec(3);
khat01 = (1/(1 - etamin) - 1)*nvec(3)/nvec(4);

if nargin == 4
    trueetatmax = 1/(1 + ((nvec(3)/nvec(4))*pi1t));
    trueetatmin = 1 - 1/(1 + ((nvec(4)/nvec(3))*pi0t));
    trueetatmaxprct = sum(trueetatmax > etat)/length(etat);
    trueetatminprct = sum(trueetatmin > etat)/length(etat);
end