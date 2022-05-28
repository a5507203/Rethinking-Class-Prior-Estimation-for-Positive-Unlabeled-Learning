function [learnbeta] = init_learnbeta(mixture_sample,component_sample,p,p1,opts)
%Creates the environment for the learnbeta function returned as output.
%INPUTS
%    mixture samples: column vector of univariate sample from mixture.
%    component samples: column vector of univariate sample from component.
%    p: The estimate of the mixture density expressed as a $k-component$ mixture;
%        %object of class mixture.
%    p1: The estimate of the component density; it is a object of a class for which 
%         pdf method is defiend, usually a mixture object or a probability
%         ditribution object.
%    opts: structure for which any of the of the folowing fields are
%        defined:
%        num_restarts (default 1): number of restarts for the optimization routine
%            
%        lossfcn: specifies the which negative log likelihood(LL) being
%            minimized. It can take the folowing values 
%              'combined': LL based on component sample and mixture sample
%              'combined2' (default) : LL based on component sample and mixture
%                  sample, with the importance of each sample controlled by
%                  opts.gamma and opts.gamma1
%              'mixture': LL based on mixture sample.
%              'component': LL based on component sample.
%        regwt (default 0): weight for the regularizer term \sum \beta_iw_i
%        consType:  
%            'eq' (default): for equality constraints \sum \beta_iw_i = c
%            'ineq': for equality constraints \sum \beta_iw_i > c
%        gamma (default 1/(number of mixture samples)): See lossfcn:'combined2'.
%        gamma1 (default 1/(number of component samples)): See lossfcn:'combined2'.
%OUTPUT
%   learnbeta: function that optimizes the log likelihood.        

    compSS = length(component_sample);
    
    DEFAULTS.num_restarts = 1;
    DEFAULTS.lossfcn = 'combined2';
    DEFAULTS.regwgt=0;
    DEFAULTS.consType='eq';
    DEFAULTS.gamma=1/length(mixture_sample);
    if compSS~=0
        DEFAULTS.gamma1=1/compSS;
    end
    if nargin < 5
        opts = DEFAULTS;
    else
        if compSS == 0
            if isfield(opts,'lossfcn')
                if ~strcmp(opts.lossfcn, 'mixture')
                    warning('Using mixture loss because component sample is not provided');
                end
            end
            opts.lossfcn='mixture';
        end
            
        opts = getOptions(opts, DEFAULTS);
    end
    
    loss_strs={'combined','combined2','component','mixture'};
    loss_fcns={@combinedLoss,@combined2Loss,@componentLoss,@mixtureLoss};
    str=validatestring(opts.lossfcn,loss_strs);
    ix=find(strcmp(str,loss_strs));
    loss_fcn=loss_fcns{ix};
    
    % Get all the components of the mixture model
    hs = p.comps;
    % avec conatins w_i 
    avec = p.mixProp'; 
    % to make sure that w sum to 1
    avec=avec/sum(avec);
    numkernels = length(avec);
    mixSS = length(mixture_sample);
    
    %to store \kappa_i values on the component sample.
    if ~strcmp(opts.lossfcn,'mixture');
        hxcomp = zeros(compSS, numkernels);
    end
    %to store \kappa_i values on the mixture sample.
    hxmix = zeros(mixSS, numkernels);
    for i = 1:numkernels
        if ~strcmp(opts.lossfcn,'mixture');
            hxcomp(:,i) = pdf(hs{i},component_sample); 
        end
        hxmix(:,i) = pdf(hs{i},mixture_sample); 
    end
    
    %wtcomp contains w_i\kappa_i evaluated on component sample.
    if ~strcmp(opts.lossfcn,'mixture');
        wtcomp = repmat(avec',compSS,1).*hxcomp;
    end
    %fhatx contains estimate of f_1 evaluted on mixture sample.
    fhatx = pdf(p1,mixture_sample);
    diffmatmix = repmat(fhatx,1,numkernels) - hxmix;
    
    % add regularizer term to the loss function.
    if opts.regwgt ~= 0
        loss_fcn = @(beta)regularized_loss(beta,loss_fcn);
    end
    
    options = optimoptions('fmincon','GradObj','on', 'Display','off');
    learnbeta = @learnbeta_fcn;
    
    
    
    function [beta,f,alpha,iter] = learnbeta_fcn(alpha_cons,beta_init)
        %minimizes the negative log likelihood.
        if nargin < 1
            alpha_cons=0;
        end
        opts.A = [-1*eye(numkernels);eye(numkernels)];
        opts.b = [zeros(numkernels,1);ones(numkernels,1)];
        % fmin_BFGS enforces Ax >= b, rather than fmincons Ax <= b
        opts.A = -1*opts.A;
        opts.b = -1*opts.b;
        alphas = zeros(1,opts.num_restarts);
        fs = zeros(1,opts.num_restarts);
        iters=zeros(1,opts.num_restarts);
        betas=zeros(numkernels,opts.num_restarts);
        for rr= 1:opts.num_restarts
            if strcmp(opts.consType,'ineq')
                opts.A = [opts.A;avec'];
                opts.b = [opts.b;alpha_cons];
                if nargin<2;
                    beta_init = rand(numkernels,1);
                    beta_init = alpha_cons + (1-alpha_cons-10^-7)*beta_init;
                end
            elseif strcmp(opts.consType,'eq')
                opts.Aeq=avec';
                opts.beq=alpha_cons;
                if nargin <2
                    beta_init =repmat(alpha_cons,numkernels,1);
                end
            end
            [beta_i,f_i,iter_i,flag] = fmincon_caller(loss_fcn,beta_init, opts);
            %[beta_i,f_i,iter_i,flag] = fmin_LBFGS(loss_fcn,beta_init, opts);
            betas(:,rr) =beta_i;
            alphas(rr)=avec'*beta_i;
            fs(rr)=f_i;
            iters(rr)=iter_i;
        end
        [f,ix_min]=min(fs);
        beta=betas(:,ix_min);
        alpha=alphas(ix_min);
        iter=iters(ix_min);
    end

%keyboard;
%%%%%%%%%%% Auxiliary functions %%%%%%%%%%%%%%%%%%%
    function [f,g] = regularized_loss(beta, mainloss)
        [f1,g1] = mainloss(beta);
        [f2,g2] = maxalpha(beta);
        f = f1+f2+f3;
        g = g1+g2+g3;  
    end
    
    function [f,g] = combinedLoss(beta)
        [f1,g1] = componentLoss(beta);
        [f2,g2] = mixtureLoss(beta);
        f = f1+f2;
        g = g1+ g2;
    end

    function [f,g] = combined2Loss(beta)
        [f1,g1] =  componentLoss(beta);
        [f2,g2] =  mixtureLoss(beta);
        f = opts.gamma1*f1+ opts.gamma*f2;
        g = opts.gamma1*g1+opts.gamma*g2;
    end
    
    function [f,g] = componentLoss(beta)
        % For all x, computes fhat(x) =
        % log(sum_i b_i a_i h_i(x)) - log(sum_i b_i a_i)
        % then returns -fhat
        [f1,g1] = lossFhat_convex(beta);
%         if strcmp(opts.consType,'ineq')
%             [f2,g2] = lossFhat_concave(beta);
%             f = f1+f2;
%             g = g1+g2;
%         elseif strcmp(opts.consType,'eq')
%             f=f1;
%             g=g1;
%         end
         [f2,g2] = lossFhat_concave(beta);
         f = f1+f2;
         g = g1+g2;
    end
    
    function [f,g] = lossFhat_concave(beta)
    % For all x, computes log(sum_i b_i a_i)
    % the concave part of lossFhat
    % log(sum_i b_i a_i)
        sbavec = sum(beta.*avec);
        f =compSS* log(sbavec);
        g = compSS* avec/sbavec;
    end
    
    function [f,g] = lossFhat_convex(beta)
    % For all x, computes -log(sum_i b_i a_i h_i(x))
    % the convex part of lossFhat
    % 1/nsamples sum_t log(sum_i b_i a_i h_i(x))
        bavec = beta.*avec;
        ksum = sum(repmat(bavec',compSS,1).*hxcomp, 2);
        ksum(ksum < 1e-5) = 1e-5;
        f = sum(log(ksum));
        g = sum(wtcomp./repmat(ksum,1,numkernels),1)';
        % The above was all for maximizing, but want to minimize
        f = -f;
        g = -g;      
    end    
    
    function [f, g] = mixtureLoss(beta)
    % For all x, computes f(x) + sum_i b_i a_i (fhat(x) - h_i(x)) 
    % = sum_i a_i h_i(x) + sum_i b_i a_i (fhat(x) - h_i(x)) = 
    % = sum_i a_i (1 - b_i) h_i(x) + sum_i b_i a_i fhat(x)
    % Get sum_i a_i (1 - b_i) h_i(x) + sum_i b_i a_i fhat(x)
        ksum = sum(repmat(((ones(numkernels,1)-beta).*avec)', ...
                          mixSS,1).*hxmix, 2) + ...
               sum(beta.*avec)*fhatx;   
        % For numerical reasons, cap ksum at some small positive value
        ksum(ksum < 1e-5) = 1e-5;
        f = sum(log(ksum));
        g = avec.*sum(diffmatmix./repmat(ksum,1,numkernels),1)';
        g = g;
        % The above was all for maximizing, but want to minimize
        f = -f;
        g = -g;
     end
    
   function [f,g] = maxalpha(beta)
    % Add the regularizer which pushes alpha to 1
    % maximize f(betavec) = <betavec,avec>, so minimize -<betavec,avec>
        f = -opts.regwgt*sum(beta.*avec);
        g = -opts.regwgt.*avec; 
   end
    
    function d= f1Dens(betavec,x)
        d= fiDens(betavec,x);
    end

    function d= f0Dens(betavec,x)
        d= fiDens(1-betavec,x);
    end
    function d= fiDens(wt,x)
        wtavec = wt.*avec;
        nx=length(x);
        hx=hFun(x);
        ksum = sum(repmat(wtavec',nx,1).*hx, 2);
        d= ksum/sum(wtavec);
    end
    function d= mixDens(betavec,x)
        hx=hFun(x);
        nx=length(x);
        fx=pdf(p1,x)';
        d= sum(repmat(((ones(numkernels,1)-betavec).*avec)', ...
                          nx,1).*hx, 2) + ...
               sum(betavec.*avec)*fx; 
    end

    function hx=hFun(x)
        nx=length(x);
        hx = zeros(nx, numkernels);
        for ii = 1:numkernels
            hx(:,ii) = pdf(hs{ii},x); % get value for h_i in the mixture
        end
    end
    
    function [beta,f,iter,flag]=fmincon_caller(loss_fcn,beta_init,opts)
        if strcmp(opts.consType,'eq')
            [beta,f,flag,output] = fmincon(loss_fcn,beta_init, -opts.A,-opts.b,opts.Aeq,opts.beq,[],[],[],options);
        elseif strcmp(opts.consType,'ineq')
            [beta,f,flag,output] = fmincon(loss_fcn,beta_init, -opts.A,-opts.b,[],[],[],options);
        end
         iter=output.iterations;
    end
end


