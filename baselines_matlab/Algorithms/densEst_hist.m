function [mixDens, comp1Dens] = densEst_hist(mixSample,comp1Sample,~)
%Estimates the density of x and x1 using finite gaussian mixtures. The
%number of components is selected using AIC. The components obtained for x1
%are resused to fit x.
set(0,'DefaultFigureVisible', 'off');
if exist('histogram') == 2
    h1=histogram(comp1Sample);
    binWidth=h1.BinWidth;
else
    [~,centers]=hist(comp1Sample);
    binWidth=centers(2)-centers(1);
end
set(0,'DefaultFigureVisible', 'off');
figure('visible','off');
xmin=min([comp1Sample;mixSample]);
xmax=max([comp1Sample;mixSample]);
binEdges=xmin:binWidth:(xmax+binWidth);
numBins=length(binEdges)-1;
binEdges=binEdges(:);
comp1Dens=toUnifMixture(repmat(1/numBins,1,numBins), binEdges(1:numBins),binEdges(2:numBins+1)); 
comp1Dens=fixedCompsFit(comp1Dens,comp1Sample);
mixDens=toUnifMixture(repmat(1/numBins,1,numBins), binEdges(1:numBins),binEdges(2:numBins+1)); 
mixDens=fixedCompsFit(mixDens,mixSample);
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