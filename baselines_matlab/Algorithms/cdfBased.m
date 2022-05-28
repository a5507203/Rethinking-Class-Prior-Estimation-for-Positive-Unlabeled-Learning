function [ alpha_est,out] = cdfBased(x,x1,opts )
%For F and F1 being smooth cdf estimates of x and x1 respectively, it finds 
%the maximum alpha such that F - alpha F1 is positive and non-decreasing. 
%This ensures that  (F - alpha F1)/(1-alpha) is a cdf as well.
    DEFAULTS.epsilon = 1e-3;
    DEFAULTS.nboots = 1;
    if nargin < 4
        opts = DEFAULTS;
    else
        opts = getOptions(opts, DEFAULTS);
    end
    component_samples=x1;
    mixture_samples=x;
    alphas=zeros(opts.nboots,1);
    for j=1:opts.nboots
        sorted_samples=sort(component_samples);
        mixcdf=ksdensity(mixture_samples,sorted_samples,'function','cdf');
        compcdf=ksdensity(component_samples,sorted_samples,'function','cdf');
        ualpha=0.999;
        lalpha=0.001;
        while(abs(ualpha-lalpha)>=0.001)
            malpha=(ualpha+lalpha)/2;
            mfnc=mixcdf - malpha * compcdf;
            %plot(sorted_samples,mfnc);
            if(checkPositive(mfnc)&&checkIncreasing(mfnc))
                lalpha=malpha;
            else
                ualpha=malpha;
            end
        end
        alpha_est=(lalpha+ualpha)/2;
        alphas(j)= alpha_est;
    end
    %boxplot(alphas);
    alpha_est=median(alphas);
    out.alpha_est=alpha_est;
    out.opts=opts;
    function ispositive=checkPositive(fnc)
        %fnc(1)=0;
        negfnc=fnc(fnc<0);
        ispositive=sum(abs(negfnc))<opts.epsilon;
    end
    function isinc=checkIncreasing(fnc)
        %fnc(1)=0;
        isinc=checkPositive(fnc(2:end)-fnc(1:end-1));
    end
end

