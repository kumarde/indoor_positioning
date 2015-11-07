% Carisa Covins
% Alan Lundgard
% Deepak Kumar
% Spencer Nofzinger
% Sudhanva Sreesha
% EECS 445 - Project
% Run Algorithm

clear all;
clc;

fprintf('Training...\n');

tic
[train_matrix, train_labels] = generate('train');
svm_models = train(train_matrix, train_labels);
toc

fprintf('Testing...\n');

tic
[test_matrix, test_labels] = generate('test');
[labels accuracy] = test(test_matrix, test_labels, svm_models);
toc

fprintf('The testing accuracy is %.2f%%\n\n', accuracy);
