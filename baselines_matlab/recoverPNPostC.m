function [pnPost,pnPost1] = recoverPNPostC(alpha,gamma, eta,alphaC, out)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
puPostC={};
pnPostC={};
pnRatioC={};
puPost1C={};
pnPost1C={};
pnRatio1C={};
global post;
global post1;
C1=out.C1;
C=out.C;
m=length(unique(C1));
pnPost=nan(length(C),1);
pnPost1=nan(length(C1),1);
%clProp=out.cSize/length(C);
%eta=(1-gamma(i)).*clProp;
for i=1:m
    temp=((1-alpha)*eta(i))/(alpha*gamma(i));
    puPostC{i}=out.xs{i};
    puPost1C{i}=out.x1s{i};
    %if(alpha==0)
    %    print("hi");
    %end
    if(~isempty(puPostC{i}))
        o=out.transforms{i};
        c=(1-o.pp)/o.pp;
        pnPostC{i}=PUpost2PNpost(puPostC{i},c,alphaC(i));
        pnRatioC{i}=PNpost2PNratio(pnPostC{i},alphaC(i));
        pnPost(C==i)=pnRatioC{i}./(pnRatioC{i}+temp);
        %if(any(isnan(pnPost(C==i))))
        %    print('hi')
        %end
        %puPost1C{i}=o.x1;
        pnPost1C{i}=PUpost2PNpost(puPost1C{i},c,alphaC(i));
        pnRatio1C{i}=PNpost2PNratio(pnPost1C{i},alphaC(i));
        pnPost1(C1==i)=pnRatio1C{i}./(pnRatio1C{i}+temp);
        %if(any(isnan(pnRatio1C{i})))
        %    print('here')
        %end    
    else
         pnPost1(C1==i)=0;
    end
end
end

