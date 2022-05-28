function [p, p1,p0,pnoisy] = gaussian_exp_pdf(locations, weights, scales,noise)
% GAUSSIAN_EXP generates a 2-component mixture of gaussians
% All parameters are optional
    if nargin < 1
        locations = [0.0, 5.0]';
    end
    [~,dim]=size(locations);
    if nargin < 2
        weights = [0.9, 0.1]';
    end
    if nargin < 3
        scales(1,:,:) = eye(dim);
        scales(2,:,:) = eye(dim);
    end
    if nargin < 4
        noise = 0;
    end
    
    
    
    
    p = tolocScaleMixture(weights,locations,scales,'Normal');  % create a mixture of 2 gaussians for testing

    p1 = tolocScaleMixture(weights(1),squeeze(locations(1,:)),squeeze(scales(1,:,:)),'Normal');
    p0 = tolocScaleMixture(weights(2),squeeze(locations(2,:)),squeeze(scales(2,:,:)),'Normal');
                
    pnoisy= mixture([1-noise,noise],{p1,p0});   
    
end