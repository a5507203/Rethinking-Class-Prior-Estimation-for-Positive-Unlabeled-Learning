function [x,d]= plotDistr( distr, npts )
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
if nargin <2
    npts=200;
end
x=random(distr,npts,1);
x=sort(x);
d=pdf(distr,x);
%plot(x,d);
end

