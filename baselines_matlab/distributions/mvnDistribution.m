classdef mvnDistribution < handle
    %Multivariate Normal distribution
    
    properties (SetAccess = private)
        mu =0;
        sigma=1;
    end
    
    methods
        function obj = mvnDistribution(mu,sigma)
            obj.mu=mu;
            obj.sigma=sigma;
        end
        function mu = get.mu(this)
            mu=this.mu;
        end
        function sigma = get.sigma(this)
            sigma=this.sigma;
        end
    end
    methods
        function dens = pdf(this,x)
            [d1,d2,d3]=size(x);
            for i = 1:d3
                dens(:,i)=mvnpdf(x(:,:,i),this.mu,this.sigma);
            end
            dens=squeeze(dens);
        end
        function s = random(this,m,n)
            for i = 1:n
                s(:,:,i)=mvnrnd(this.mu,this.sigma,m);
            end
            s=squeeze(s);
        end
   end
end

