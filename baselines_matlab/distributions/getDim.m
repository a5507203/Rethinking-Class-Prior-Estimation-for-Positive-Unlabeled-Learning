function [ dim ] = getDim(distr)
%get dimension of a distribution by sampling from it.
dim=numel(squeeze(distr.random(1,1)));

end

