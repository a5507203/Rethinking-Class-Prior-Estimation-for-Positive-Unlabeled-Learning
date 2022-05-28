function [pnPost,pnPost1] = recoverPNPost(alpha,out)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here

o=out.transform;

puPost=o.x;
c=(1-o.pp)/o.pp;
pnPost=PUpost2PNpost(puPost,c,alpha);
puPost1=o.x1;
pnPost1=PUpost2PNpost(puPost1,c,alpha);

end


