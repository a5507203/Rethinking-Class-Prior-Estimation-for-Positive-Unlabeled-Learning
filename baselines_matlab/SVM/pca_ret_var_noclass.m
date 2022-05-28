function [rpm, proj] = pca_ret_var_noclass(data, ret_var)

% PCA_RET_VAR    Eigenvector projection (arbitrary dimension)
%
%   d        - pattern matrix (data)
%   num_feat - number fo feature columns
%   ret_var  - desired retained variance
%   rpm      - returns projected rotated centered pattern matrix
%   proj    - projection matrix
%
% NOTE: THIS FUNCTION ASSUMES THAT THE DATASET HAS BEEN NORMALIZED
% TO HAVE ZERO MEAN.

    %data = d(:, 1 : num_feat);

    % compute covariance matrix and perform eigenanalysis
    [vecs, vals] = eig(cov(data));

    vals  = diag(vals); % vals was a square matrix 
    total = sum(vals);  % calculate total variance in this data

    % sort eigenvalues in descending order
    [sorted_vals, index] = sort(vals, 'descend');

    % find dimensionality such that retained variance is below required
    q = find(cumsum(sorted_vals) / total < ret_var / 100);
    
    % in case all features are strictly co-linear, ret_var < 100 will
    % lead to an empty q; in that case we must take the first PCA feature
    if isempty(q)
        dim = 1;
    else
        dim = q(length(q));
    end
    
    % form projection matrix. First coord has highest variance, etc.
    proj = vecs(:, index(1 : dim));

    % project matrix
    rpm = data * proj;
    
    % add class label column(s)
    %rpm = [rpm(:, 1 : dim) d(:, (num_feat + 1) : size(d, 2))];
return

