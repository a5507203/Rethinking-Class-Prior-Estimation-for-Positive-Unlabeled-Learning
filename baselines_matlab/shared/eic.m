function [v] = eic(c)
%if c is a cell it exposes it else leave it as it is
if iscell(c)
    v=c{:};
else
    v=c;
end
end

