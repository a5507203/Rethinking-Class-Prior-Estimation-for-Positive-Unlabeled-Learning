function [alpha_ests,out] = pdfRatio(x,x1,opts)
%Estimates the mixing proportion from the density estimates f and f1 corresponding 
%to samples x and x1 respectively. The minimum value of the ratio f/f1 is the
%estimate. A robust estimate is also computed by taking the median of
% some percent of smallest ratios.
DEF.percentile= 0.1;
    if nargin < 3;
        opts=DEF;
    else
        opts=getOptions(opts,DEF);
    end
    f=ksdensity(x,x1);
    f1=ksdensity(x1,x1);
    ratio = f./f1;
    ratio=sort(ratio);
    alpha_ests(1)=ratio(1);
    
    nx1=length(x1);
    max_ix=round(opts.percentile*nx1);
    alpha_ests(2)=median(ratio(1:max_ix));
    out.alpha_ests=alpha_ests;
    out.opts=opts;
end