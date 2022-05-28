function [ cons ] = logscale_cons(lower_limit,upper_limit, num_cons)
%generates num_cons values between lower_limit and upper_limit. The values
%are equally spaced in log scale.
l_log=log(lower_limit);
u_log=log(upper_limit);
sep=(u_log-l_log)/num_cons;
cons=exp(l_log:sep:u_log);
cons=cons(1:end-1);
end

