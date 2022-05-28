function [alpha_est,out]= inflectionScript(alphas,fs,opts)
     %fitting two lines with a constraint that the lines are below the points   
        DEF.npts=10; % minimum number of points required to fit a line
        DEF.k=5;
        if nargin < 3
            opts = DEF;
        else
            opts = getOptions(opts, DEF);
        end
        npts=opts.npts;
        fs=fs(:);
        alphas=alphas(:);
        
        indna=isnan(fs);
        fs=fs(~indna);
        alphas=alphas(~indna);
        
        [alphas,srt_ix]=sort(alphas);
        fs=fs(srt_ix);
        
        fs=estimate_fs(alphas);
        fs= (fs - min(fs))/(max(fs)-min(fs));
        n=length(alphas);
        srs=zeros(n,1);
        beta1s=nan(2,n);
        beta2s=nan(2,n);
        X=[ones(n,1),alphas];
        min_slope= (fs(end)-fs(1))/(alphas(end)-alphas(1))/100    ;
        options = optimoptions('quadprog','Display','off');
        for j = npts:n-npts
            Xj=X(j+1:j+npts,:);
            Qj= Xj'*Xj;
            bj= -fs(j+1:j+npts,:)'*Xj;
            beta1=quadprog(Qj,bj,[X;0,-1],[fs;-min_slope],[],[],[],[],[],options);
            beta1s(:,j)=beta1;
            beta2s(:,j+npts)=beta1;
            if isnan(beta2s(1,j))
                Xj=X(j-npts+1:j,:);
                Qj= Xj'*Xj;
                bj= -fs(j-npts+1:j,:)'*Xj;
                beta2=quadprog(Qj,bj,[X;0,-1],[fs;-min_slope],[],[],[],[],[],options);
                beta2s(:,j)=beta2;
            end
            sr=(beta1(2)-beta2(2))/(median(fs(j-npts+1:j))+10^-2);
            srs(j)=sr;
        end
        [~,ix]=max(srs);
        alpha_est=alphas(ix);
        out.alpha_est=alpha_est;
        out.coeff1=beta1s(:,ix);
        out.coeff2=beta2s(:,ix);
    
    function fs_est=estimate_fs(alphas1)
        IDX=knnsearch(alphas,alphas1,'K',opts.k);
        fs_est=median(fs(IDX),2);
    end
            
end

