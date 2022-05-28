function [alpha_u ] = alpha_unlabeledData(alpha_c, ss_x,ss_x1,beta)
%transforms proportion of positives in combined dataset to the proportion
%of positives in the unlabeled dataset
if nargin < 4
    beta=1
end
alpha_u= (alpha_c*(ss_x+ss_x1) - beta*ss_x1)/ss_x ;
end

