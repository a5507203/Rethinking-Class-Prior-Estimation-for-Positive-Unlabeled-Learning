% USAGE:
%	[theta, PE] = PUpepriorest(xp, xm, sigma_list, lambda_list)
% Estimates the class prior of the dataset xm using the positive labeled samples
%	xp by partial matching of the distributions.
%
% INPUT:
% 	xp: The positive labeled samples
%	xm: The mixture of samples from the positive and negative classes with
%		and unknown class prior
%	sigma_list: A list of candidate sigma values. For default, a sigma list
%		around the median distance is used.
%	lambda_list: A list of lambda parameter
%	kfolds: The number of folds for cross validation (default 5)
%
% OUTPUT:
%	prior: The class prior p(y=1)
%	ym: The labelling for the xm dataset
%	t: prior range (used for plotting)
%	PE: Pearson divergence for values in t
%	IOprior: The prior for the inlier-outlier setting
%
% NOTE: This is the version used in the paper. For large datasets,
%  fix the number of basis functions.
%
function [prior, ym, t, PE] = PEPriorEst(xp, xm, sigma_list, lambda_list, kfolds)

	[d1, n1] = size(xp);
	[d2, n2] = size(xm);
	%disp(d1)
	%disp(n1)
	%disp(d2)
	%disp(n2)
	% set the default sigma list
	if (nargin<4)
		 [~, med_dist] = CalculateDist2([xp xm], [xp xm]);
		 sigma_list = linspace(1/5, 5, 10)*med_dist;
	end

	% set the lambda list
	if (nargin<5)
		lambda_list=logspace(-3,1,9); 
	end

	% set the number of folds
	if (nargin<6)
		kfolds = 5;
	end

	% randomly permute the data
	xp = xp(:, randperm(n1));
	xm = xm(:, randperm(n2));
	
	% split the data into training and test folds
	cv_split_xp = floor([0:n1-1]*kfolds./n1)+1;
	cv_split_xm = floor([0:n2-1]*kfolds./n2)+1;

	% set the cv scores
	cv_scores = zeros(length(sigma_list), length(lambda_list));

	for k=1:kfolds

		% get the training and test datasets
		idx_xp_tr = find(cv_split_xp~=k);
		idx_xp_te = find(cv_split_xp==k);

		idx_xm_tr = find(cv_split_xm~=k);
		idx_xm_te = find(cv_split_xm==k);
		
		n1_tr = length(idx_xp_tr);
		n1_te = length(idx_xp_te);
		
		n2_tr = length(idx_xm_tr);
		n2_te = length(idx_xm_te);

		p1_tr = n1_tr/(n1_tr+n2_tr);
		p1_te = n1_te/(n1_te+n2_te);

		xp_tr = xp(:, idx_xp_tr);
		xp_te = xp(:, idx_xp_te);

		xm_tr = xm(:, idx_xm_tr);
		xm_te = xm(:, idx_xm_te);

		% set the basis functions (numerator without test samples)
		x_ce = [xp_tr];
		b = size(x_ce, 2);

		for sigma_idx = 1:length(sigma_list)
			sigma = sigma_list(sigma_idx);

			% Calculate the Phi matrices
			Phi1_tr = GaussBasis(xp_tr, x_ce, sigma)';
			Phi1_te = GaussBasis(xp_te, x_ce, sigma)';

			Phi2_tr = GaussBasis(xm_tr, x_ce, sigma)';
			Phi2_te = GaussBasis(xm_te, x_ce, sigma)';

			% calculate H and h
			h_tr = mean(Phi1_tr, 2);
			h_te = mean(Phi1_te, 2);

			H_tr = p1_tr*Phi1_tr*Phi1_tr'./n1_tr + (1-p1_tr)*Phi2_tr*Phi2_tr'./n2_tr;
			H_te = p1_te*Phi1_te*Phi1_te'./n1_te + (1-p1_te)*Phi2_te*Phi2_te'./n2_te;
	
			for lambda_idx = 1:length(lambda_list)
				lambda = lambda_list(lambda_idx);
				
				% calculate the alpha values
				alpha = linsolve(H_tr + lambda*eye(b), h_tr);

				% calculate the score
				score = 1/2*alpha'*H_te*alpha - alpha'*h_te;
				
				cv_scores(sigma_idx, lambda_idx) = cv_scores(sigma_idx, lambda_idx) + score;
			end % lambda
		end % sigma
	end %k
	clear('sigma', 'lambda', 'h_tr', 'H_tr');

	% select the hyper-parameter that minimizes the CV score
	[~, idxp, idxm] = min2(cv_scores);
	sigma_chosen = sigma_list(idxp); 
	lambda_chosen = lambda_list(idxm);

	% setup the basis functions and calculate Phi, h, and H
	x_ce = [xp];
	b = size(x_ce, 2);
	Phi1 = GaussBasis(xp, x_ce, sigma_chosen)';
	Phi2 = GaussBasis(xm, x_ce, sigma_chosen)';

	p1 = n1./(n1+n2);
	
	h = mean(Phi1, 2);
	H = p1*Phi1*Phi1'/n1 + (1-p1)*Phi2*Phi2'/n2;
	
	% calculate the density ratio
	alpha = linsolve(H + lambda_chosen*eye(b), h);
	
	% calculate the prior
	prior = (2*alpha'*h - alpha'*H*alpha)^(-1);
	
	prior = max(prior, n1/(n1+n2));
	prior = min(prior, 1);
	
	% calculate the labeling for each xm
	pxm = alpha'*Phi1;
	ym = zeros(size(pxm));
	ym(pxm>0.5) = 1;

	% calculate the values for the PE at certain t values
	t = linspace(0, 1, 1000);
	PE = t.^2*(alpha'*h - 1/2*alpha'*H*alpha) - t + 1/2;	
end

% Stupid little function that is quite useful
function [mval, idxp, idxm] = min2(A)

	mval = inf;
	idxp = 1;
	idxm = 1;

	for i=1:size(A,1)
		for j=1:size(A,2)
			if (A(i,j)<mval)
				mval = A(i,j);
				idxp = i;
				idxm = j;
			end
		end
	end
end


% Gaussian basis function
function [Phi] = GaussBasis(z, c, sigma)
% from LSIF code
	Phi_tmp=-(repmat(sum(c.^2,1),size(z,2),1) ...
	+repmat(sum(z.^2,1)',1,size(c,2))-2*z'*c)/2;

	Phi=exp(Phi_tmp/(sigma^2));
end

% Calculates the squared distance between the column vectors in matrix X and the column vectors in matrix Y.
% X -> input matrix of dxn
% Y -> matrix of centers, dxm
% XC_dist2 -> output matrix of size nxm
% med_dist -> the median distance between the vectors
function [XC_dist2, med_dist] = CalculateDist2(X, Y)

	[d nx]=size(X);
	[d nc]=size(Y);
	X2=sum(X.^2,1);
	Y2=sum(Y.^2,1);

	% calculate the square
	XC_dist2 = repmat(Y2, nx, 1)+repmat(X2',1,nc)-2*X'*Y;

	% calculate the median distance
	if (nargout>1)
		med_dist = sqrt(median(XC_dist2(:)));
	end
end
