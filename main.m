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

SET = 'one';
BINS = 50;
NORMALIZATION = 2;

tic
[train_matrix train_labels] = generate(SET, 'train');
kd_tree_models = train(train_matrix);
toc

fprintf('\n\nTesting...\n');

tic
[test_matrix test_labels] = generate(SET, 'test');
[labels accuracy] = test( ...
	train_matrix,         ...
	train_labels,         ...
	test_matrix,          ...
	test_labels,          ...
	kd_tree_models);
toc

fprintf('The testing accuracy is %.2f%%\n\n', accuracy);
