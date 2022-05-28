function [w] = weighted_logreg (X, y, c)

% Weighted logistic regression, maximizes weighted likelihood
% X = n-by-k data matrix of reals
% y = n-by-1 vector of zeros and ones, class labels
% c = n-by-1 vector of costs, all costs must be positive

max_step = 100;     % maximum number of iterations
tolerance = 0.01;   % threshold of change in weights (between iterations)

% add a column of ones to matrix X
X = [ones(size(X, 1), 1) X];

% initial coefficients using ordinary least squares regression
w = inv(X' * diag(c) * X) * X' * diag(c) * y;

% calculate log-likelihood
ll = get_log_likelihood_w(X, y, w, c);

step = 1;
eps = tolerance;

% updates will stop if the number of steps exceeds some maximum number
% or if the relative change of w is smaller than a prespecified number
% or if the log likelihood is too close to zero (classes are separable)
while step <= max_step & eps >= tolerance & ll < -1e-6
    % vector of posterior probabilities that class equals 1
    p = logsig(X * w);
    
    % store w vector, normalized
    w_old = w / sum(w);
    
    % apply update rule and get new w
    w = w + inv(X' * diag(c .* p .* (1 - p)) * X) * X'* diag(c) * (y - p);
    
    % percent difference between old and new (normalized) w vectors
    eps = sum(abs(w_old - w / sum(abs(w))));
    
    % calculate log-likelihood
    ll = get_log_likelihood_w(X, y, w, c);
    
    step = step + 1;
end

return
