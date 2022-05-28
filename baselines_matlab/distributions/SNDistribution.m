classdef SNDistribution < handle
    %UNTITLED5 Summary of this class goes here
    %   Detailed explanation goes here
    
    properties (SetAccess = private)
        mu =0;
        sigma=1;
        lambda=0;
    end
    
    methods
        function obj = SNDistribution(mu,sigma,lambda)
            obj.mu=mu;
            obj.sigma=sigma;
            obj.lambda=lambda;
        end
        function mu = get.mu(this)
            mu=this.mu;
        end
        function sigma = get.sigma(this)
            sigma=this.sigma;
        end
        function lambda = get.lambda(this)
            lambda=this.lambda;
        end
    end
    methods
        function dens = pdf(this,x)
            [nSample,dim]=size(x);
            z=(x-this.mu)/this.sigma;
            dens=2/this.sigma *normpdf(z).*normcdf(this.lambda*z);
        end
        function s = random(this,m,n)
            deltaSq=this.lambda^2/(1+this.lambda^2);
            s=this.mu + sign(this.lambda)*this.sigma*sqrt(deltaSq)*abs(normrnd(0,1,m,n)) + ...
                sqrt(1-deltaSq)*normrnd(0,1,m,n);           
        end
   end
end