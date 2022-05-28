classdef mixture < handle
    %UNTITLED5 Summary of this class goes here
    %   Detailed explanation goes here
    
    properties (SetAccess = private)
        mixProp =[];
        comps=cell(1,0);
    end
    
    methods
        function obj = mixture(mixProp,comps)
            obj.mixProp=mixProp;
            obj.comps=comps;
        end
        function mixProp = get.mixProp(this)
            mixProp=this.mixProp;
        end
        function comps = get.comps(this)
            comps=this.comps;
        end
    end
    methods
        function dens = pdf(this,x)
            [nSample,dim]=size(x);
            dens= zeros(nSample,1);
            for i = 1: length(this.mixProp)
                dens = dens + this.mixProp(i)*pdf(this.comps{i},x);
            end
        end
         function [sample,label] = random(this,m,n)
            ncomps=length(this.mixProp);
            sampleSize=m*n;
            dim=getDim(this.comps{1});
            s=zeros(sampleSize,dim);
            compSS=round(sampleSize*this.mixProp);
            compSS(ncomps)=compSS(ncomps)+ sampleSize -sum(compSS);
            ind=0;
            for i = 1:ncomps 
                lb(ind+1:ind+compSS(i))=i;
                s(ind+1:ind+compSS(i),:)=random(this.comps{i},compSS(i),1);
                ind=ind+compSS(i);
            end
            ind=randperm(sampleSize);
            s=s(ind,:);
            lb=lb(ind);
            %ind2sub([m,n],ind);
            sample=nan(m,dim,n);
            label=nan(m,n);
            for j=1:n
                label(:,j)=lb(1+(j-1)*m:j*m);
                sample(:,:,j)=s(1+(j-1)*m:j*m,:);
            end
         end
         function mix = fixedCompsFit(this,x,niter)
            if(nargin<3)
                niter=1000;
            end
            hx=compwisePdf(this, x);
            dim=size(hx);
            npts=dim(1);
            for j=1:niter
                tt= hx.*repmat(this.mixProp,npts,1);
              
                dd=sum(tt,2)+10^-9;
                %plot(x,dd);
                tt= tt./repmat(dd,1,length(this.mixProp));
                if any(isnan(tt));
                    error('nan tuty');
                end
                this.mixProp=mean(tt,1);
                this.mixProp=this.mixProp/sum(this.mixProp);
                if any(isnan(this.mixProp));
                    error('nan mixProp');
                end
                %av=av+10^-4;
                %av=av/sum(av);
            end
            this.mixProp=this.mixProp/sum(this.mixProp);
            mix=this;
         end
         function compPdfs = compwisePdf(this,x)
             dim=size(x);
             npts=dim(1);
             ncomps=length(this.comps);
             compPdfs=zeros(npts,ncomps);
             for j=1:ncomps
                compPdfs(1:npts,j)=pdf(this.comps{j},x);
            end
         end
        function c = cdf(this,x)
             [nSample,dim]=size(x);
            c= zeros(nSample,1);
            for i = 1: length(this.mixProp)
                c = c + this.mixProp(i)*cdf(this.comps{i},x);
            end
        end
        function this = remove0(this)
            positive=this.mixProp>0; 
            this.mixProp=this.mixProp(positive);
            this.comps=this.comps(positive);
        end
   end
end

