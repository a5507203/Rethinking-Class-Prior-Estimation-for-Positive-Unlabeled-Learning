classdef laplaceDistribution < handle
    %Laplace Distribution
    
    properties (SetAccess = private)
        mu =0;
        sigma=1;
    end
    
    methods
        function obj = laplaceDistribution(mu,sigma)
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
            [nSample,dim]=size(x);
            dens=1/(2*this.sigma) *exp(-abs(x-this.mu)/this.sigma);
        end
        function s = random(this,m,n)
            s=laprnd(m,n,this.mu,this.sigma);            
        end
   end
end