function [alpha, out ] = gmm(x,x1,opts)
    DEF.max_iter=1000;
    DEF.thr=10^-6;
    DEF.rr=1;
if nargin < 3
    opts=DEF;
end
opts=getOptions(opts,DEF);
n1=length(x1);
n=length(x);
[m1_p,s1_p]=ml_est(x1);
alphas=rand(opts.rr,1);
fits=cell(length(alphas),1);
rr=length(alphas);
ix=1;
for r=1:rr;
    al_p = alphas(r);
   [m0_p,s0_p]=init_param0(al_p,m1_p,s1_p); 
   al=0;[m1,s1]=ml_est(x);[m0,s0]=ml_est(x);
   l_p=ll(al_p,m1_p,s1_p,m0_p,s0_p);l=ll(al,m1,s1,m0,s0);
   k=1;
   ls=nan(opts.max_iter,1);
   iter=1;
   while true
       dx_1=normal_density(m1_p,s1_p);
       dx_0=normal_density(m0_p,s0_p);
       d1_x=al_p*dx_1./(al_p*dx_1+(1-al_p)*dx_0);
       d0_x= 1 - d1_x;
       al_into_n=sum(d1_x);
       al = al_into_n/n;
       m1=m1_p;
       m1 = sum(x.*d1_x)/al_into_n;
       m0= sum(x.*d0_x)/(n-al_into_n);
       s1= sum(d1_x.*((x-m1).^2))/al_into_n;
       s1=s1_p;
       s0= sum(d0_x.*((x-m0).^2))/(n-al_into_n);
       l=ll(al,m1,s1,m0,s0);
       ls(iter)=l;
       if criteria(k, [al,m1,s1,m0,s0,l],[al_p,m1_p,s1_p,m0_p,s0_p,l_p])
           break;
       end
       al_p=al;m1_p=m1;s1_p=s1;m0_p=m0;s0_p=s0;l_p=l;
       iter=iter+1;
   end 
   %scatter(1:length(ls),ls);
   param.mu1=m1;param.alpha=al;param.s1=s1;param.mu0=m0;param.s0=s0;
   param.iter=iter;
   fits{r}=param;
   if better(fits{r},fits{ix})
       ix=r;
   end
end
out.ll=ll(fits{ix}.alpha,fits{ix}.mu1,fits{ix}.s1,fits{ix}.mu0,fits{ix}.s0) ;
if abs(mean(x1)-fits{ix}.mu1) < abs(mean(x1)-fits{ix}.mu0)
    alpha=fits{ix}.alpha;
    out.param=fits{ix};
else
    alpha=1-fits{ix}.alpha;
    out.param.mu1=fits{ix}.mu0; out.param.s1=fits{ix}.s0;
    out.param.mu0=fits{ix}.mu1; out.param.s0=fits{ix}.s1;
    out.param.alpha=alpha;
end
out.fits=fits;
out.alpha=alpha;
    function [mu,sig]=ml_est(sample)
        mu=mean(sample);
        sig=sum((sample-mu).^2)/length(sample);
    end
    function [mu,sig]=init_param0(alpha,m,s)
         n_al=round(n*alpha);
         p1=makedist('Normal','mu',m,'sigma',s);
         sample1=random(p1,n_al,1);
         cix= setdiff(1:length(x), dsearchn(x,sample1));
         [mu,sig] =ml_est(x(cix));
    end
    function stop = criteria(k,val_new,val_old)
        stop = k>opts.max_iter; 
        change=sum(abs(val_new-val_old)./abs(val_new+val_old)/2);
        stop = stop | change < opts.thr;
    end
    function d = normal_density(m,s)
        if(isnan(m))
            error('hi');
        end
      p=makedist('Normal','mu',m,'sigma',sqrt(s));
        d=max(pdf(p,x),10^-10);
    end
    function l=ll(al,m1,s1,m0,s0)
        d=al*normal_density(m1,s1)+(1-al)*normal_density(m0,s0);
        l=mean(log(d));
    end
    function b=better(new,best)
        b=ll(new.alpha,new.mu1,new.s1,new.mu0,new.s0) > ...
            ll(best.alpha,best.mu1,best.s1,best.mu0,best.s0) ;
    end
end

