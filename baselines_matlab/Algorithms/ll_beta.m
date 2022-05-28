function [ ll_cmb,ll_mix,ll_cmp,ll_cmb2,out] = ll_beta(beta,x,x1,mix_dens,comp_dens,opts)
%Computes the combined data log likelihood with for a beta vector that
%rewights the components of the mixture.
DEF.gamma=1/length(x);
DEF.gamma1=1/length(x1);
if nargin < 6
    opts=DEF;
else
    opts=getOptions(opts,DEF);
end
as=mix_dens.mixProp';
alpha=sum(beta.*as);
out.h0s=h0(x);
out.h1s=h1(x);
out.hs=h(x);
ll_mix=sum(log(h(x)));
if ~isempty(x1)
    ll_cmp=sum(log(h1(x1)));
    ll_cmb=ll_cmp+ll_mix;
    ll_cmb2=ll_cmp*opts.gamma1+ll_mix*opts.gamma;
else
    ll_cmp=nan;ll_cmb=nan;ll_cmb2=nan;
end
    
    function h_val=h(xx)
        h_val=alpha * comp_dens.pdf(xx) + (1-alpha) * h0(xx);
    end
    function hh1_val=hh1(xx,bt)
        new_mp= bt.*as;
        new_mp=new_mp/sum(new_mp);
        new_mix=mixture(new_mp,mix_dens.comps);
        hh1_val= max(new_mix.pdf(xx),10^-14); 
    end
    function h1_val=h1(xx)
        h1_val=hh1(xx,beta);
    end
    
    function h0_val=h0(xx)
        h0_val=hh1(xx,1-beta);
    end
end

