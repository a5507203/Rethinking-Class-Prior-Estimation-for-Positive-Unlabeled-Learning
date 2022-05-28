function [ dist ] = makedistWrapper(distname, varargin)
%Creates Normal and Laplace distribtuion objects. 
%Multivariate Laplace not supported yet

if(strcmpi(distname,'Laplace'))
    dist=laplaceDistribution(varargin{2},varargin{4});
elseif strcmpi(distname,'Gaussian') 
    if length(varargin{2})==1
        dist=makedist('normal',varargin{:});
    else
        dist=mvnDistribution(varargin{2},varargin{4});
    end
elseif strcmpi(distname,'Uniform') 
    if length(varargin{2})==1
        dist=makedist(distname,varargin{:});
    else
        dist=mvUniformDistribution(varargin{2},varargin{4});
    end
end

end

