function [strs,ix]= multi_strmatch(strs1,strs2)
% searches all the strngs from strs1 in strs2 and returns back those for
% which match was found

ix=isnan(1:length(strs2));
for jx = 1:length(strs1)
    ix=ix|strcmp(strs1{jx},strs2);
end
strs=strs2(ix);
end