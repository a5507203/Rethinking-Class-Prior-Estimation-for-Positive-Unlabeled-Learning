function b = n_fold (len, n)

q = randperm(len);

for i = 1 : n
    b{i} = q(i : n : len);
end

return


