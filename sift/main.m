% Carisa Covins
% Alan Lundgard
% Deepak Kumar
% Spencer Nofzinger
% Sudhanva Sreesha
% EECS 445 - Project
% Run Algorithm

clear all;
clc;

run('../toolbox/vlfeat-0.9.20/toolbox/vl_setup.m');

fprintf('Training...\n');

SET = 'one';

tic
[train_matrix train_labels] = preprocess(SET, 'train');
kd_tree_models = train(train_matrix);
toc

fprintf('\n\nValidating...\n');

tic
[validation_matirx validation_matirx] = preprocess(SET, 'test');
[labels accuracy] = validate( ...
	train_matrix,             ...
	train_labels,             ...
	validation_matirx,        ...
	validation_matirx,        ...
	kd_tree_models);
toc

fprintf('SIFT with k-d Trees'' validation accuracy is %.2f%%\n\n', accuracy);
