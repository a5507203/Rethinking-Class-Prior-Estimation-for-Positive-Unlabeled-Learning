function [estimate, output] = AlphaMax_noisy(x,x1_noisy,opts)
%AlphaMax for x1 with noise
%%
DEF.est_ind=1;
DEF.type='Corrected';
if nargin < 3
    opts=DEF;
else
    opts=getOptions(opts,DEF);
end
%%
%To obtain the estimate of component 1 density.
[alphas,fs,out]=compute_llCurve(x,x1_noisy,opts);
fs=llCurve_correction(alphas,fs);
est=inflectionScript(alphas,-fs);
output.AM1=out;
output.AM1.alphas=alphas;
output.AM1.fs=fs;
output.AM1.est=est;
au_plus=est(opts.est_ind);

[alphas,fs,out]=compute_llCurve(x1_noisy,x,opts);
fs=llCurve_correction(alphas,fs);
est=inflectionScript(alphas,-fs);
output.AM2=out;
output.AM2.alphas=alphas;
output.AM2.fs=fs;
output.AM2.est=est;
al_plus=1-est(opts.est_ind);

output.au_star= au_plus*al_plus/(1-au_plus*(1-al_plus));
output.al_star= al_plus/(1-au_plus*(1-al_plus));
estimate=output.au_star;
end

