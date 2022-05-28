classdef mvUniformDistribution < handle
    %Laplace Distribution
    
    properties (SetAccess = private)
        lb = 0;
        ub = 1;
    end
    
    methods
        function obj = mvUniformDistribution(lb,ub)
            obj.lb=lb;
            obj.ub=ub;
        end
        function lb = get.lb(this)
            lb=this.lb;
        end
        function ub = get.ub(this)
            ub=this.ub;
        end
    end
    methods
        function dens = pdf(this,x)
            del=this.ub-this.lb;
            if ~all(del>0)
                error('incorrect parameter for a mutivariate uniform distribution');
            end
            [ss,~]=size(x);
            above_lb = x-repmat(this.lb,ss,1) > 0 ;
            below_ub = repmat(this.ub,ss,1)-x >= 0 ;
            dens=repmat(1/prod(del),ss,1);
            dens=prod(above_lb,2).*prod(below_ub,2).*dens;
        end
        function s = random(this,m,n)
            for d = 1:length(this.ub)
                s(:,d,:)=random('Uniform',this.lb(d),this.ub(d),m,n);
            end
            s=squeeze(s);
        end
   end
end
