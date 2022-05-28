function [g,g1,p,p1] = cdfGaussTransform(x,x1)
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here
v=[x;x1];
z = [zeros(length(x),1);ones(length(x1),1)];
[~,ix]=sort(v);
z_srt=z(ix);
%cdf values
pr=cumsum(z_srt)/sum(z_srt==1);
%gaussian transform
gs=sqrt(2)*erfinv(2*pr-1);

gs=handleInf(gs,z_srt==1);
gs=handleInf(gs,z_srt==0);
g1=gs(z_srt==1);
g=gs(z_srt==0);
p1=pr(z_srt==1);
p=pr(z_srt==0);

    function x = handleInf(x,ix)
        if(sum(isfinite(x)&ix)==0)
            mx=max(x(isfinite(x)));
            mn= min(x(isfinite(x)));
        else
            mx=max(x(isfinite(x)&x));
            mn= min(x(isfinite(x)&x));
        end
        x(ix&x==inf)=mx;
        x(ix&x==-inf)=mn;
    end
end

