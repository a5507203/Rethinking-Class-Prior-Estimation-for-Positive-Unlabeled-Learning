function mix = toUnifMixture(mixProp,lower, upper)
        k=length(mixProp);
        comps=cell(1,k);
        for j =1:k
           comps{j}=makedistWrapper('Uniform','Lower', squeeze(lower(j,:)),'Upper',squeeze(upper(j,:)));
        end
        mix=mixture(mixProp, comps);
 end