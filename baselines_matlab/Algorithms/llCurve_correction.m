function    fs = llCurve_correction(alphas,fs)
%Corrects the log likelihood curve by enforcing it to be non increasing
for i=1:length(fs)
    ai=alphas<alphas(i);
    fs(ai)=max(fs(ai),fs(i));
end
end

