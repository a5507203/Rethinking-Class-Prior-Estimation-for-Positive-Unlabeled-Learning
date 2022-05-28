function [ v ] = ift(strct, fld)
%tells if the structure has fiels fld and the value of field is true
if isfield(strct,fld)
    v=strct.(fld);
else
    v=false;
end
end

