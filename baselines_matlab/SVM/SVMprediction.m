function [labels_test_predicted] = SVMprediction (bigtrain, bigtest, info)

% determine number of features
n_features = size(bigtrain, 2) - 1;

% change labels from 0 to -1 to adjust to SVM toolbox
labels_train(find(bigtrain(:, size(bigtrain, 2)) == 1)) = 1;
labels_train(find(bigtrain(:, size(bigtrain, 2)) == 0)) = -1;
labels_train = labels_train';

file1 = random_filename('file1', '');
file2 = random_filename('file2', '');
file3 = random_filename('file3', '');
file4 = random_filename('file4', '');

% determine cost ratio for the SVM toolbox (we want to adjust learning for
% good balanced accuracy)
ratio = length(find(labels_train == -1)) / length(find(labels_train == 1));
%ratio = 1;
options = svmlopt('Regression', 0, 'Kernel', info.kernel, 'KernelParam', info.parameter, 'CostFactor', ratio * info.pos_weight, 'ExecPath', info.SVMlightpath);
% train SVM - warning is turned off since toolbox outputs some "errors"
% which are not important
warning off
svmlwrite(file1, bigtrain(:, 1 : n_features), labels_train);
status = svm_learn(options, file1, file2);
if status ~= 0
    error('SVMlight svm_learn did not work properly');
end
svmlwrite(file3, bigtest(:, 1 : n_features));
status = svm_classify(svmlopt(options), file3, file2, file4);
if status ~= 0
    error('SVMlight svm_classify did not work properly');
end
labels_test_predicted = svmlread(file4);
warning on

delete(file1);
delete(file2);
delete(file3);
delete(file4);

return
