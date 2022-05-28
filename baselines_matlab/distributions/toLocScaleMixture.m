 function [mix,comps] = toLocScaleMixture(mixProp,distr,mu, Sigma)
        k=length(mixProp);
        comps=cell(1,k);
        for j =1:k
            comps{j}=makedistWrapper(distr,'mu', squeeze(mu(j,:)),'sigma',squeeze(Sigma(:,:,j)));
        end
        mix=mixture(mixProp, comps);
 end