% Carisa Covins
% Alan Lundgard
% Deepak Kumar
% Spencer Nofzinger
% Sudhanva Sreesha
% EECS 445 - Project
% Using code from mini project, using RGB
% images of 96*96*3 dimension for CNN.

initW = 1e-2;
initB = 1e-1;

opts.continue = false;
opts.gpus = [];

opts.learningRate = 1e-2;
opts.batchSize = 128;
opts.numEpochs = 30;

opts.expDir = fullfile('tmp', 'convnet');
if exist(opts.expDir, 'dir') ~= 7, mkdir(opts.expDir); end

addpath('../data/');
addpath('../toolbox/cnn');
addpath(genpath('../toolbox/matconvnet-1.0-beta16'));
addpath('../toolbox/matconvnet-1.0-beta16/examples');
addpath('../toolbox/matconvnet-1.0-beta16/matlab');
%rmpath('../toolbox/matconvnet-1.0-beta16/matlab/vl_nnpool');

load('tmp/train.mat');
load('tmp/val.mat');

nTrain = size(trainData, 1);
nVal = size(valData, 1);


train_data_normalized_standardized = trainData;
val_data_normalized_standardized = valData;
[~, imLength] = size(trainData);

for i = 1 : nTrain
    for color = 1 : 3
        avg = mean(trainData(i, (color-1)*96*96+1:color*96*96));
        tempData = trainData(i, (color-1)*96*96+1:color*96*96) - avg;
        normResult = norm(tempData);
        train_data_normalized_standardized(i, (color-1)*96*96+1:color*96*96) = tempData/normResult*10;
    end
end
for i = 1 : nVal
    for color = 1:3
        avg = mean(valData(i, (color-1)*96*96+1:color*96*96));
        tempData = valData(i, (color-1)*96*96+1:color*96*96) - avg;
        normResult = norm(tempData);
        val_data_normalized_standardized(i, (color-1)*96*96+1:color*96*96) = tempData/normResult*10;
    end
end
disp('normalizing done');
for i = 1 : imLength
    meanVal = mean(train_data_normalized_standardized(:, i));
    tempData = train_data_normalized_standardized(:, i) - meanVal;
    stdDev = std(tempData);
    train_data_normalized_standardized(:, i) = tempData / stdDev;
    val_data_normalized_standardized(:, i) = (val_data_normalized_standardized(:, i) - meanVal) / stdDev;
end
disp('standardizing done');

% Set up architecture
initConvnetModelLarge.layers = {};
initConvnetModelLarge.layers{end + 1} = struct('type', 'conv', ...
    'weights', {{initW*randn(5,5,3,32, 'single'), ...
    initB*randn(1,32, 'single')}}, ...
    'stride', 2, ...
    'pad', 2);
initConvnetModelLarge.layers{end + 1} = struct('type', 'relu');
initConvnetModelLarge.layers{end + 1} = struct('type', 'pool', ...
    'method', 'max', ...
    'pool', [3, 3], ...
    'stride', 2, ...
    'pad', 1);
initConvnetModelLarge.layers{end + 1} = struct('type', 'conv', ...
    'weights', {{initW*randn(5,5,32,64, 'single'), ...
    initB*randn(1,64, 'single')}}, ...
    'stride', 1, ...
    'pad', 2);
initConvnetModelLarge.layers{end + 1} = struct('type', 'relu');
initConvnetModelLarge.layers{end + 1} = struct('type', 'pool', ...
    'method', 'max', ...
    'pool', [3, 3], ...
    'stride', 2, ...
    'pad', 1);
initConvnetModelLarge.layers{end + 1} = struct('type', 'conv', ...
    'weights', {{initW*randn(5,5,64,64, 'single'), ...
    initB*randn(1,64, 'single')}}, ...
    'stride', 1, ...
    'pad', 2);
initConvnetModelLarge.layers{end + 1} = struct('type', 'relu');
initConvnetModelLarge.layers{end + 1} = struct('type', 'pool', ...
    'method', 'max', ...
    'pool', [3, 3], ...
    'stride', 2, ...
    'pad', 1);
initConvnetModelLarge.layers{end + 1} = struct('type', 'conv', ...
    'weights', {{initW*randn(6,6,64,4096, 'single'), ...
    initB*randn(1,4096, 'single')}}, ...
    'stride', 1, ...
    'pad', 0);
initConvnetModelLarge.layers{end + 1} = struct('type', 'relu');
initConvnetModelLarge.layers{end + 1} = struct('type', 'conv', ...
    'weights', {{initW*randn(1,1,4096,8, 'single'), ...
    initB*randn(1,8, 'single')}}, ...
    'stride', 1, ...
    'pad', 0);
initConvnetModelLarge.layers{end+1} = struct('type', 'softmaxloss');


convnetModel = cnnTrain(train_data_normalized_standardized,...
                            trainLabels, ...
                            val_data_normalized_standardized,...
                            valLabels,...
                            initConvnetModelLarge,...
                            opts);

%%
% Predict on the validation data
convnetValPredictionLarge = cnnPredict(val_data_normalized_standardized, convnetModel, opts);
convnetValAccuracyLarge = mean(valLabels == convnetValPredictionLarge)*100;
fprintf('Convolutional neural network validation accuracy: %g%%\n', convnetValAccuracyLarge);

% Get the iteration-error plot
copyfile(fullfile(opts.expDir, 'net-train.pdf'), fullfile('tmp', 'rgb.pdf'));


