function auc = get_auc_ultra (s, y)

% function auc = get_auc_ultra (s, y)
%
% Calculates area under the ROC curve for a binary classifier.
%
% Inputs:
%         s:   a vector of prediction scores; it is assumed that higher 
%              scores correspond to higher y's
%         y:   a vector of binary class values (0 or 1 only)
%
% Outputs:
%         auc: area under the ROC curve; a number in [0, 1] interval
%
% Predrag Radivojac
% Indiana University
% Bloomington, IN 47405
% U.S.A.
% October 2013

if ~isnumeric(s) || ~isnumeric(y)
    error('Incorrect input arguments: s and y must be numeric.')
end

if ~isvector(s) || ~isvector(y)
    error('Incorrect input arguments: s and y must be vectors.')
end

[sr sc] = size(s);
[yr yc] = size(y);
if sr ~= yr || sc ~= yc
    if sr == yc && sc == yr
        if yr == 1
            y = y';
        else
            s = s';
        end
    else
        error('Incorrect input arguments: s and y must have the same dimensions.')
    end
end

uy = unique(y);
if length(uy) ~= 2 || uy(1) ~= 0 || uy(2) ~= 1
    error('Data contains only one class, calculating AUC makes no sense.')
end
    
% calculate auc for ascending order of y
[y r] = sort(y, 'ascend');
s = s(r);
auc = fast_auc(s, y);

% calculate auc for descending order of y and then average two auc's
[y r] = sort(y, 'descend');
s = s(r);
auc = (auc + fast_auc(s, y)) / 2;

return


function auc = fast_auc (score, y)

n = length(y);              % number of data points
n0 = length(find(y == 0));  % number of negatives
n1 = n - n0;                % number of positives

% sort y in descending order
[~, b] = sort(score, 'descend');
y = y(b);


% find all places in the sorted array where the class changes
q = find(y(1 : n - 1) ~= y(2 : n));

% obtain differential values
q = [q; n] - [0; q];

% set the sizes for true positive rate (tpr) and false positive rate (fpr)
tpr = zeros(1, length(q) + 1);
fpr = zeros(1, length(q) + 1);

if y(1) == 1
    x = cumsum(q(1 : 2 : length(q))) / n1;
    tpr(2 : 2 : length(tpr)) = x;
    if 3 + 2 * (length(x) - 1) > length(tpr)
        tpr(3 : 2 : length(tpr)) = x(1 : length(x) - 1);
    else
        tpr(3 : 2 : length(tpr)) = x;
    end
    
    x = cumsum(q(2 : 2 : length(q))) / n0;
    fpr(3 : 2 : length(fpr)) = x;
    if 4 + 2 * (length(x) - 1) > length(fpr)
        fpr(4 : 2 : length(fpr)) = x(1 : length(x) - 1);
    else
        fpr(4 : 2 : length(fpr)) = x;
    end
else
    x = cumsum(q(2 : 2 : length(q))) / n1;
    tpr(3 : 2 : length(tpr)) = x;
    if 4 + 2 * (length(x) - 1) > length(tpr)
        tpr(4 : 2 : length(tpr)) = x(1 : length(x) - 1);
    else
        tpr(4 : 2 : length(tpr)) = x;
    end
    
    x = cumsum(q(1 : 2 : length(q))) / n0;
    fpr(2 : 2 : length(fpr)) = x;
    if 3 + 2 * (length(x) - 1) > length(fpr)
        fpr(3 : 2 : length(fpr)) = x(1 : length(x) - 1);
    else
        fpr(3 : 2 : length(fpr)) = x;
    end
end

auc = trapz(fpr, tpr);

%plot(fpr, tpr)

return
