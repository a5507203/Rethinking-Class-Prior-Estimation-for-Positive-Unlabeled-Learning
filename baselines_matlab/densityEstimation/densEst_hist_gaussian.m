function [mixDens, comp1Dens] = densEst_hist_gaussian(mixSample,comp1Sample,~)
%Estimates the density of x and x1 using finite gaussian mixtures. The
%number of components is selected using AIC. The components obtained for x1
%are resused to fit x.
set(0,'DefaultFigureVisible', 'off');
if exist('histogram') == 2
    h1=histogram(mixSample);
    binWidth=h1.BinWidth;
else
    [~,centers]=hist(mixSample);
    binWidth=centers(2)-centers(1);
end
set(0,'DefaultFigureVisible', 'off');
figure('visible','off');
%xmin=min(comp1Sample);
%xmax=max(comp1Sample);
xmin=min(mixSample);
xmax=max(mixSample);
binEdges=xmin:(binWidth*2):(xmax+binWidth);
%xminMix=min(mixSample);
%xmaxMix=max(mixSample);
%if(binEdges(1)>xminMix)
%   binEdges = [xminMix,binEdges]; 
%end
%if(binEdges(end)<xmaxMix)
%   binEdges = [binEdges,xmaxMix]; 
%end
binEdges=binEdges(:);
numBins=length(binEdges)-1;
mixDens=toUnifMixture(repmat(1/numBins,1,numBins), binEdges(1:numBins),binEdges(2:numBins+1)); 
mixDens=fixedCompsFit(mixDens,mixSample);
comp1Dens=makedist('Normal');
%x=sort(mixSample);
%x1=sort(comp1Sample);
%hold off
%plot(x,pdf(mixDist,x));
  %   hold on;     
%plot(x1,pdf(comp1Dist,x1));
%comp1Dist=fixedCompsFit(comp1Dist,x1);
%plot(x,pdf(comp1Dist,x));
     %hold off;
end
