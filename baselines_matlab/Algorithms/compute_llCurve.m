function [alphas, fs, out] = compute_llCurve(x,x1,opts)
%computes the log likelihood versus alpha(mixing proportion)
DEF.densityEst_fcn=@densEst_hist;
DEF.constraints = 0.01:0.01:0.99;
DEF.consType='eq';
DEF.parallel=false;
DEF.num_restarts=1;
DEF.gamma=1/length(x);
DEF.gamma1=1/length(x1);
DEF.loss_str='combined2';
if nargin < 3
    opts=DEF;
else
    opts=getOptions(opts,DEF);
end
if ~isfield(opts,'dens');
    [dens.p,dens.p1]=opts.densityEst_fcn(x,x1);
else
    dens=opts.dens;
end
loss_strs={'combined','combined2','component','mixture'};
ll_strs={'ll_cmb','ll_cmb2','ll_cmp','ll_mix'};
str=validatestring(opts.loss_str,loss_strs);
jx=find(strcmp(str,loss_strs));
ll_str=ll_strs{jx};

out.dens=dens;
numkernels=length(dens.p.mixProp);
opts1.num_restarts=opts.num_restarts;
opts1.lossfcn=opts.loss_str;
opts1.consType=opts.consType;
opts1.gamma=opts.gamma;
opts1.gamma1=opts.gamma1;

learnbeta=init_learnbeta(x,x1,dens.p,dens.p1,opts1);

cons=opts.constraints;
num_lbs=length(cons);
alphas=zeros(1,num_lbs);
ll_cmb=nan(1,num_lbs);
ll_cmp=nan(1,num_lbs);
ll_mix=nan(1,num_lbs);
ll_cmb2=nan(1,num_lbs);
fs=nan(1,num_lbs);
objs=nan(1,num_lbs);
iters=zeros(1,num_lbs);
betas=nan(numkernels,num_lbs);
init=nan(numkernels,num_lbs);
ll_bt= @(beta)ll_beta(beta,x,x1,dens.p,dens.p1,opts);
if(opts.parallel)
    parfor k = 1:num_lbs
        [beta,o,alpha,iter] = feval(learnbeta,cons(k));
        betas(:,k) =beta;
        if(isnan(o))
            error('objective nan');
        end
        alphas(k)=alpha;
        objs(k)=o;
        [ll_cmb(k),ll_mix(k),ll_cmp(k),ll_cmb2(k)]=ll_bt(beta);
        iters(k)=iter;
    end
else
    for k = 1:num_lbs
        beta_init =get_beta_init(k);
        if any(isnan(beta_init))
            [beta,o,alpha,iter] = feval(learnbeta,cons(k));
        else
            try
                   [beta,o,alpha,iter] = feval(learnbeta,cons(k),beta_init);
            catch
                warning('optimization failed')
            end
        end
        if(isnan(o))
            error('objective nan');
        end
        betas(:,k) =beta';
        alphas(k)=alpha;
        objs(k)=o;
        [ll_cmb(k),ll_mix(k),ll_cmp(k),ll_cmb2(k)]=ll_bt(beta);
        iters(k)=iter;
        update_init(k,beta);
    end
end
fs=eval(ll_str);
out.objs=objs;
out.betas=betas;
out.iters=iters;
out.ll_cmb=ll_cmb;out.ll_mix=ll_mix;out.ll_cmp=ll_cmp;out.ll_cmb2=ll_cmb2;
out.alphas=alphas;
out.fs=fs;

    function beta_init=get_beta_init(kk)
        beta_init=init(:,kk);
    end
    function update_init(kk,bt)
        if kk < num_lbs
            bt=min(1-10^-8,bt);
            bt=max(10^-8,bt);
            if strcmp(opts.consType,'eq')
                while abs(cons(kk+1)-sum(bt.*dens.p.mixProp')) >1e-12;
                    c=cons(kk+1)/sum(bt.*dens.p.mixProp');
                    bt=min(1-10^-8,c*bt);
                end
            elseif strcmp(opts.consType,'ineq')
                bt=min(1-10^-8,bt);
                bt=max(10^-8,bt);
                while sum(bt.*dens.p.mixProp')-cons(kk+1) < 1e-4;
                    c=(cons(kk+1)+1e-3)/sum(bt.*dens.p.mixProp');
                    bt=min(1-10^-8,c*bt);
                end
            end
            init(:,kk+1)=bt;
        end
    end
end

