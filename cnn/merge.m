% Carisa Covins
% Alan Lundgard
% Deepak Kumar
% Spencer Nofzinger
% Sudhanva Sreesha
% EECS 445 - Project
% After each room has its images in bbb.mat, dow.mat, etc. we need to
% combine them into train.mat. As mentioned earlier, had to break up each
% room when loading into .mat files because of the size. Possible to
% combine here because of rescaling down.

fprintf('Running merge.m...\n');

load('tmp/bbb');
BBB_size = size(train_labels, 1);
load('tmp/dow');
dow_size = size(train_labels, 1);
load('tmp/eecs');
eecs_size = size(train_labels, 1);
load('tmp/dude');
dude_size = size(train_labels, 1);
load('tmp/fxb');
fxb_size = size(train_labels, 1);
load('tmp/hall');
hall_size = size(train_labels, 1);
load('tmp/name');
name_size = size(train_labels, 1);
load('tmp/pier');
pier_size = size(train_labels, 1);

%{
% Using 500 as the amount of photos from each building/room
BBB_size = 1;
dow_size = BBB_size;
fxb_size = dow_size;
eecs_size = dow_size;
dude_size = dow_size;
name_size = dow_size;
pier_size = dow_size;
hall_size = dow_size;
%}

% We assume each building.mat is in /tmp
nTrain = BBB_size + dow_size + eecs_size + dude_size + fxb_size + hall_size + name_size + pier_size;
trainData = zeros(nTrain, 96*96*3);
trainLabels = zeros(nTrain, 1);
sum = 0;
load('tmp/bbb');
for i = 1 : BBB_size
	trainData(i, :) = reshape(imresize(reshape(train_data(i, :), 612, 816, 3),[96, 96]), 1, 96*96*3);
	trainLabels(i + sum, :) = 1;
end
sum = sum + BBB_size;
load('tmp/dow');
for i = 1 : dow_size
	trainData(i + sum, :) = reshape(imresize(reshape(train_data(i, :), 612, 816, 3),[96, 96]), 1, 96*96*3);
	trainLabels(i + sum, :) = 2;
end
sum = sum + dow_size;
load('tmp/eecs');
for i = 1 : eecs_size
	trainData(i + sum, :) = reshape(imresize(reshape(train_data(i, :), 612, 816, 3),[96, 96]), 1, 96*96*3);
	trainLabels(i + sum, :) = 3;
end
sum = sum + eecs_size;
load('tmp/dude');
for i = 1 : dude_size
	trainData(i + sum, :) = reshape(imresize(reshape(train_data(i, :), 612, 816, 3),[96, 96]), 1, 96*96*3);
	trainLabels(i + sum, :) = 4;
end
sum = sum + dude_size;
load('tmp/fxb');
for i = 1 : fxb_size
	trainData(i + sum, :) = reshape(imresize(reshape(train_data(i, :), 612, 816, 3),[96, 96]), 1, 96*96*3);
	trainLabels(i + sum, :) = 5;
end
sum = sum + fxb_size;
load('tmp/hall');
for i = 1 : hall_size
	trainData(i + sum, :) = reshape(imresize(reshape(train_data(i, :), 612, 816, 3),[96, 96]), 1, 96*96*3);
	trainLabels(i + sum, :) = 6;
end
sum = sum + hall_size;
load('tmp/name');
for i = 1 : name_size
	trainData(i + sum, :) = reshape(imresize(reshape(train_data(i, :), 612, 816, 3),[96, 96]), 1, 96*96*3);
	trainLabels(i + sum, :) = 7;
end
sum = sum + name_size;
load('tmp/pier');
for i = 1 : pier_size
	trainData(i + sum, :) = reshape(imresize(reshape(train_data(i, :), 612, 816, 3),[96, 96]), 1, 96*96*3);
	trainLabels(i + sum, :) = 8;
end
sum = sum + pier_size;

save(fullfile('tmp', 'train.mat'), 'trainData', 'trainLabels');

% Also reshape validation data here too, for consistency
load('tmp/val.mat');
[nVal, ~] = size(val_data);
valData = zeros(nVal, 96*96*3);
for i = 1 : nVal
	valData(i, :) = reshape(imresize(reshape(val_data(i, :), 612, 816, 3),[96,96]), 1, 96*96*3);
end
valData = double(valData);
valLabels = double(val_labels);

save(fullfile('tmp', 'val.mat'), 'valData', 'valLabels');

fprintf('merge.m finished\n');

clear;
