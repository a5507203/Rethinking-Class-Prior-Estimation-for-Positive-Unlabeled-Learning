function [XX,YY] = binaryWaveform(n0,n1)
%Generate Waveform dataset with binary class attribute

N = round(max(3/2 *n1, 3*n0))+3;
X = generate_waveform(N); % the last 3 columns are 3 binary classes
y = zeros(size(X, 1), 1); % prepare vector y
q1 = find(X(:, 22) == 1); % find examples from classes 1-3
q2 = find(X(:, 23) == 1);
q3 = find(X(:, 24) == 1);
X = X(:, 1 : 21);         % keep only the features of the data matrix

% convert to binary classification problem
% classes 2 and 3 will become class 1, class 1 becomes class 0
% classes 2 and 3 combined give a non-linear concept for bin. class.
y([q3; q2], 1) = 1;

% now, we need to reduce the data set to match n0 and n1 (N was 10xn)
q0 = find(y == 0);
q1 = find(y == 1);
q0 = q0(1 : n0);   % the code will crash here if N was not large enough
q1 = q1(1 : n1);   % if it doesn't, all is great
X0=X(q0,:);
X1=X(q1,:);
XX=[X0;X1];
YY=[y(q0);y(q1)];
end

