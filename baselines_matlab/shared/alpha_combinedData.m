function [alpha_c ] = alpha_combinedData(alpha_u, ss_x,ss_x1)
%transforms proportion of positives in unlabeled dataset to the proportion
%of positives in the entire dataset
alpha_c= (alpha_u*ss_x + ss_x1)/(ss_x+ss_x1)
end

