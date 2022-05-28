function varargout = ballinbox(d, nPos, nNeg, errPos, errNeg, method, boxEdgeHalfLength)
% function ballinbox generate two classes of instances in a 2*unit box,
% with positive class inside the unit ball and negative class outside
% input:
%   d (default = 3): dimensionality (>1 when boxEdgeHalfLength = 1, >=1
%       when boxEdgeHalfLength > 1) of the instances
%   nPos (default = 1000): number of positive instances
%   nNeg (default = 1000): number of negative instances to generate
%   errPos (default = 0): a real number from 0 to 1, is the percentage of wrong label for
%       the positive class
%   errNeg (default = 0): a real number from 0 to 1, is the percentage of wrong label for
%       the positive class
%   method (default = 0): 0 - uniform volume,1 -uniform radius, method used to generate positive instances 
%   boxEdgeHalfLength (default = 1): half length of the box's edge
% output:
%   x: n*d matrix for the n = nPos + nNeg instances
%   t: n*1 vector with class labels, 1 for positive, 0 for negative
% usage:
%   [x, t] = ballinbox(d, nPos, nNeg,...);
%   xt = ballinbox(d, nPos, nNeg,...);
% Yong Fuga Li, yonli@umail.iu.edu
% Oct.28, 2010
% Nov.1, 2010, add boxEdgeHalfLength as an input

if ~exist('errPos','var'), errPos = 0;end;
if ~exist('errNeg','var'), errNeg = 0;end;
if ~exist('method','var'), method = 0;end;
if ~exist('d','var'), d = 3;end;
if ~exist('boxEdgeHalfLength','var'), boxEdgeHalfLength = 1;end
if (d == 1 && boxEdgeHalfLength == 1) || d < 1 || boxEdgeHalfLength < 1
    error('Dimensionality d and boxEdgeHalfLength should not be smaller than 1, or both be 1!');
end

errPos = round(nPos*errPos);
errNeg = round(nPos*errNeg);
t = [ones(nPos-errPos,1); zeros(nNeg-errNeg+errPos,1); ones(errNeg,1)];
x = [randball(d,nPos,method); randbox(d,nNeg,boxEdgeHalfLength)];
r = randperm(nPos + nNeg); % do permutation;
x = x(r,:);
t = t(r);
if nargout == 1
    varargout{1} = [x t];
elseif nargout == 2
    varargout{1} = x;
    varargout{2} = t;
else
end

function x = randbox(d,n,boxEdgeHalfLength,r)
% generate random instances in 2*unit box with dots outside the unit ball
if ~exist('r','var')
    r = 1;
end
x = (2*rand(n,d)-1)*boxEdgeHalfLength;
torepair = sum(x.^2,2) < r;
nErr = sum(torepair);
if nErr
    x(torepair,:) = randbox(d,nErr,boxEdgeHalfLength,r);
end

