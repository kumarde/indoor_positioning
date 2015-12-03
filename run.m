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

SET = 'two';
BINS = 50;
NORMALIZATION = 2;

tic
[train_matrix train_labels] = generate(SET, 'train', BINS, NORMALIZATION);
svm_models = train(train_matrix, train_labels);
toc

fprintf('\n\nTesting...\n');

tic
[test_matrix test_labels] = generate(SET, 'train', BINS, NORMALIZATION);
[labels accuracy] = test(test_matrix, test_labels, svm_models);
toc

fprintf('The testing accuracy is %.2f%%\n\n', accuracy);
