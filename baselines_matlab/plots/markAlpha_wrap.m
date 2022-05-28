function markAlpha_wrap(est,alphas,fs)
%Wrapper for mark alpha that uses max of fs as the height of the marking
%line
ix=knnsearch(alphas(:),est);
set(gca,'Xlim',[0,1]);
mark_alpha(est,struct('col','k','is_vert',true,'max_val',fs(ix),'lStyle','--'))
end

