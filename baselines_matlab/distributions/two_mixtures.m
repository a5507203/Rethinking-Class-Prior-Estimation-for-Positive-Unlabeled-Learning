function [mix1, mix2, comps] = two_mixtures(comp_distr,locations, scales, weights1, weights2)
% generates two 2-component Laplace/Gaussian mixtures
% All parameters are optional
     if nargin < 1
        locations = [0.0, 5.0]';
    end
    [ncomps,dim]=size(locations);
    if nargin < 2
        weights1 = repmat(1/ncomps,ncomps,1);
    end
    if nargin < 3
        for i=1:ncomps
            scales(:,:,i) = eye(dim);
        end
    end
    if nargin < 4
        weights2 = [1;zeros(ncomps-1,1)];
    end
    
    [mix1,comps]=toLocScaleMixture(weights1,comp_distr,locations,scales);
    mix2= mixture(weights2,comps);   

%     if nargin < 4
%         noise = 0;
%     end
%     if nargin < 3
%         scales = ones(1,length(weigths));
%     end
%     if nargin < 2
%         weights = [0.1, 0.9];
%     end
%     if nargin < 1
%         locations = [0.0, 5.0];
%     end
%     p = tolocScaleMixture(weights,locations,scales,'laplace');
%         
%     p1 = tolocScaleMixture(weights(1),locations(1),scales(1),'laplace');
%     p0 = tolocScaleMixture(weights(2),locations(2),scales(2),'laplace');
%                 
%     pnoisy= mixture([1-noise,noise],{p1,p0});   

end