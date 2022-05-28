function filename = random_filename (name, extension)

% function filename = random_filename (name, extension)
%
% Returns a random filename based on a system clock
%
% Example:
%
% f = random_filename('file', 'txt')
%
% returns
%
% file11_14_2002_10_47_1175.txt
%
% if the system clock was set to 11/14/2002, 10:47am and 11.75 seconds
%
% Predrag Radivojac, 2002

tm = clock;

filename = [name num2str(tm(2)) '_' num2str(tm(3)) '_' num2str(tm(1))  '_' num2str(tm(4)) '_' num2str(tm(5)) '_'...
        num2str(round(tm(6) * 100)) '.' extension];

return