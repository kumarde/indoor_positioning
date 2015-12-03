%%

% EECS 445, Fall 2015
% Mini project: facial expression classification

% Release: 5pm, Oct 27, 2015
% Due: 5pm, Nov 24, 2015

%% 0 Get started
% Switch the current working directory to the folder which this script is in
% (suppose this script is in your root directory).
cd(fileparts(mfilename('fullpath')));
% Put "training.mat", "validation.mat" and "test.mat" into the folder "data/",
% otherwise you will get error information in the command window
assert(exist(fullfile('data', 'training.mat'), 'file') > 0, 'training.mat not found.');
assert(exist(fullfile('data', 'validation.mat'), 'file') > 0, 'validation.mat not found.');
assert(exist(fullfile('data', 'test.mat'), 'file') > 0, 'test.mat not found.');
% Create a folder "result" for saving results to report
if ~exist('result', 'dir'), mkdir('result'); end
% Create a folder "tmp" for saving intermediate results
if ~exist('tmp', 'dir'), mkdir('tmp'); end
% Clean up the workspace, figures and the command window
clear; close all; clc;

%% 1 Softmax and linear SVM classifiers on hand-crafted features (20 points)

% Load data from files
load(fullfile('data', 'training.mat'));
% "trainData" and  "trainLabels" loaded
% trainData: 28709x2304, each row is a training data point
% trainLabels: 28709x1, each row is the label (0-6) of a training data point
% Use 1-7 to denote the labels instead of 0-6
if min(trainLabels(:)) == 0, trainLabels = trainLabels+1; end

load(fullfile('data', 'validation.mat'));
% "valData" and "valLabels" loaded
% valData: 3589x2304, each row is a validation data point
% valLabels: 3589x1, each row is the label (0-6) of a validation data point
% Use 1-7 to denote the labels instead of 0-6
if min(valLabels(:)) == 0, valLabels = valLabels+1; end

load(fullfile('data', 'test.mat'));
% "testData" loaded: 3589x2304, each row is a test data point

%% 1.a Data cleaning: remove bad samples (6 points)

% Identify blank (pure colored) images in training/validation data
% Find out the indexes (row vectors "badTrainSampleIndex" and "badValSampleIndex")
% of the rows for those bad samples

%-------------------------------------------------------------------
%       Your code here (replace the code when necessary)
badTrainSampleIndex = [];
[N, ~] = size(trainData);
color = mean(trainData, 2);
for i = 1 : N
    if norm(trainData(i, :) - color(i)) < 1
        badTrainSampleIndex = [badTrainSampleIndex i];
    end
end
fprintf('Bad training sample indexes: '); disp(badTrainSampleIndex);
badValSampleIndex = [];
[N, ~] = size(valData);
color = mean(valData, 2);
for i = 1 : N
    if norm(valData(i, :) - color(i)) < 1
        badValSampleIndex = [badValSampleIndex i];
    end
end
fprintf('Bad validation sample indexes: '); disp(badValSampleIndex);
badTestSampleIndex = [];
[N, ~] = size(testData);
color = mean(testData, 2);
for i = 1 : N
    if norm(testData(i, :) - color(i)) < 1
        badTestSampleIndex = [badTestSampleIndex i];
    end
end
fprintf('Bad test sample indexes: '); disp(badTestSampleIndex);
%--------------------------------------------------------------------

% Remove the rows correspond to bad training/validation samples
trainData(badTrainSampleIndex, :) = [];
trainLabels(badTrainSampleIndex) = [];
valData(badValSampleIndex, :) = [];
valLabels(badValSampleIndex) = [];
testData(badTestSampleIndex, :) = [];

nTrain = size(trainData, 1); % the # of training data points
nVal = size(valData, 1); % the # of validation data points
nTest = size(testData, 1); % the # of test data points

% Save the indexes of bad training/validation samples to files
save(fullfile('result', 'badTrainSampleIndex.mat'), 'badTrainSampleIndex');
save(fullfile('result', 'badValSampleIndex.mat'), 'badValSampleIndex');
save(fullfile('result', 'badTestSampleIndex.mat'), 'badTestSampleIndex');

%% 1.b Hand-crafted feature extraction: HoG and LBP features (8 points)

% Extract HoG and LBP features by Vlfeat toolbox (http://www.vlfeat.org/)
% Put the decompressed files to "toolbox0/vlfeat-0.9.20"
% You may need to recompile the mex libraries if you are not working in
% Windows. Please follow the instructions on the official website for
% compilation.

addpath(genpath(fullfile('toolbox0', 'vlfeat-0.9.20', 'toolbox', 'mex')));

% HoG (cell size = 8): function vl_hog (use the command "help vl_hog" to check the argument formats)
% LBP (cell size = 8): function vl_lbp (use the command "help vl_lbp" to check the argument formats)

%-------------------------------------------------------------------
%       Your code here (replace the code when necessary)

hogFeatureDimension = (48/8)*(48/8)*31;
trainHogFeatures = zeros(nTrain, hogFeatureDimension);
valHogFeatures = zeros(nVal, hogFeatureDimension);
testHogFeatures = zeros(nTest, hogFeatureDimension);
side = 48; % length/width of image
for i = 1 : nTrain
    trainHogFeatures(i, :) = reshape(vl_hog(single(reshape(trainData(i, :), side, side)), 8), 1, hogFeatureDimension);
end
for i = 1 : nVal
    valHogFeatures(i, :) = reshape(vl_hog(single(reshape(valData(i, :), side, side)), 8), 1, hogFeatureDimension);
end
for i = 1 : nTest
    testHogFeatures(i, :) = reshape(vl_hog(single(reshape(testData(i, :), side, side)), 8), 1, hogFeatureDimension); 
end


lbpFeatureDimension = (48/8)*(48/8)*58;
trainLbpFeatures = zeros(nTrain, lbpFeatureDimension);
valLbpFeatures = zeros(nVal, lbpFeatureDimension);
testLbpFeatures = zeros(nTest, lbpFeatureDimension);
for i = 1 : nTrain
    trainLbpFeatures(i, :) = reshape(vl_lbp(single(reshape(trainData(i, :), side, side)), 8), 1, lbpFeatureDimension);
end
for i = 1 : nVal
    valLbpFeatures(i, :) = reshape(vl_lbp(single(reshape(valData(i, :), side, side)), 8), 1, lbpFeatureDimension);
end
for i = 1 : nTest
    testLbpFeatures(i, :) = reshape(vl_lbp(single(reshape(testData(i, :), side, side)), 8), 1, lbpFeatureDimension); 
end

%--------------------------------------------------------------------

% Save the extracted HoG and LBP features to files
save(fullfile('tmp', 'hogFeatures.mat'), 'trainHogFeatures', 'valHogFeatures', 'testHogFeatures');
save(fullfile('tmp', 'lbpFeatures.mat'), 'trainLbpFeatures', 'valLbpFeatures', 'testLbpFeatures');

%% 1.c Classification with Softmax classifier (3 points)

% Perform classification using Softmax toolbox (in "toolbox0/softmax")
% This Softmax implementation is speeded up by the L-BFGS optimization method
addpath(genpath(fullfile('toolbox0', 'softmax')));

% Train Softmax: function softmaxTrain (use the command "help softmaxTrain" to check the argument formats)

%-------------------------------------------------------------------
%       Your code here (replace the code when necessary)
softmaxModelOnHog = softmaxTrain(trainHogFeatures.', trainLabels);
softmaxModelOnLbp = softmaxTrain(trainLbpFeatures.', trainLabels);
%--------------------------------------------------------------------

% Predict on the validation data
softmaxValPredictionOnHog = softmaxPredict(softmaxModelOnHog, valHogFeatures');
softmaxValAccuracyOnHog = mean(valLabels == softmaxValPredictionOnHog')*100;
fprintf('Softmax (on HoG) validation accuracy: %g%%\n', softmaxValAccuracyOnHog);
softmaxTestPredictionOnHog = softmaxPredict(softmaxModelOnHog, testHogFeatures');

softmaxValPredictionOnLbp = softmaxPredict(softmaxModelOnLbp, valLbpFeatures');
softmaxValAccuracyOnLbp = mean(valLabels == softmaxValPredictionOnLbp')*100;
fprintf('Softmax (on Lbp) validation accuracy: %g%%\n', softmaxValAccuracyOnLbp);
softmaxTestPredictionOnLbp = softmaxPredict(softmaxModelOnLbp, testLbpFeatures');

% Save the model and predictions files
save(fullfile('tmp', 'softmaxModelOnHog.mat'), 'softmaxModelOnHog');
save(fullfile('result', 'softmaxOnHog.mat'), 'softmaxValPredictionOnHog', ...
    'softmaxTestPredictionOnHog', 'softmaxValAccuracyOnHog');
save(fullfile('tmp', 'softmaxModelOnLbp.mat'), 'softmaxModelOnLbp');
save(fullfile('result', 'softmaxOnLbp.mat'), 'softmaxValPredictionOnLbp', ...
    'softmaxTestPredictionOnLbp', 'softmaxValAccuracyOnLbp');

%% 1.d Classification with linear SVM classifiers (3 points)

% Perform classification using LibLinear toolbox (https://www.csie.ntu.edu.tw/~cjlin/liblinear/)
% Put the decompressed files to "toolbox0/liblinear-2.1"
% You may need to recompile the mex libraries if you are not working in
% Windows. You can use the compiled libraries that you used in homework 3.
% Please follow the instructions on the official website for compilation.

addpath(genpath(fullfile('toolbox0', 'liblinear-2.1')));

% Train linear SVM: function train (use the command "train" to check the argument formats)
% Use the solver "L2-regularized L2-loss support vector classification (primal)"
% (add "-s 2" option) to speed up training

%-------------------------------------------------------------------
%       Your code here (replace the code when necessary)

linearSvmModelOnHog = train(trainLabels, sparse(trainHogFeatures), '-s 2');
linearSvmModelOnLbp = train(trainLabels, sparse(trainLbpFeatures), '-s 2');
%-------------------------------------------------------------------

% Predict on the validation data
linearSvmValPredictionOnHog = predict(valLabels, sparse(valHogFeatures), linearSvmModelOnHog, '-q');
linearSvmValAccuracyOnHog = mean(valLabels == linearSvmValPredictionOnHog)*100;
fprintf('Linear SVM (on HoG) validation accuracy: %g%%\n', linearSvmValAccuracyOnHog);
linearSvmTestPredictionOnHog = predict(zeros(nTest, 1), sparse(testHogFeatures), linearSvmModelOnHog, '-q');

linearSvmValPredictionOnLbp = predict(valLabels, sparse(valLbpFeatures), linearSvmModelOnLbp);
linearSvmValAccuracyOnLbp = mean(valLabels == linearSvmValPredictionOnLbp)*100;
fprintf('Linear SVM (on LBP) validation accuracy: %g%%\n', linearSvmValAccuracyOnLbp);
linearSvmTestPredictionOnLbp = predict(zeros(nTest, 1), sparse(testLbpFeatures), linearSvmModelOnLbp, '-q');

% Save the model and predictions files
save(fullfile('tmp', 'linearSvmModelOnHog.mat'), 'linearSvmModelOnHog');
save(fullfile('result', 'linearSvmOnHog.mat'), 'linearSvmValPredictionOnHog', ...
'linearSvmValAccuracyOnHog', 'linearSvmTestPredictionOnHog');
save(fullfile('tmp', 'linearSvmModelOnLbp.mat'), 'linearSvmModelOnLbp');
save(fullfile('result', 'linearSvmOnLbp.mat'), 'linearSvmValPredictionOnLbp', ...
    'linearSvmValAccuracyOnLbp', 'linearSvmTestPredictionOnLbp');

%% 2 A lightweight convolutional neural networks (30 points)

% Clean up variables that are no longer used to save memory
clear trainHogFeatures valHogFeatures trainLbpFeatures valLbpFeatures ...
    testHogFeatures testLbpFeatures;
clear softmaxModelOnHog softmaxModelOnLbp softmaxValPredictionOnHog ...
    softmaxTestPredictionOnHog softmaxValPredictionOnLbp softmaxTestPredictionOnLbp;
clear linearSvmModelOnHog linearSvmModelOnLbp linearSvmPredictionOnHog linearSvmPredictionOnLbp;

%% 2.a Data pre-processing: downscaling, normalization and standardization (15 points)

% Downscale the images from 48x48 to 32x32 using bicubic interpolation (5 point)
% Useful Matlab function: imresize

%-------------------------------------------------------------------
%       Your code here (replace the code when necessary)
trainData32x32 = zeros(nTrain, 32*32);
valData32x32 = zeros(nVal, 32*32);
testData32x32 = zeros(nTest, 32*32);
for i = 1 : nTrain
    trainData32x32(i, :) = reshape(imresize(reshape(trainData(i, :), 48, 48), [32, 32]), 1, 32*32); 
end
for i = 1 : nVal
    valData32x32(i, :) = reshape(imresize(reshape(valData(i, :), 48, 48), [32, 32]), 1, 32*32); 
end
for i = 1 : nTest
    testData32x32(i, :) = reshape(imresize(reshape(testData(i, :), 48, 48), [32, 32]), 1, 32*32); 
end
%--------------------------------------------------------------------

% Save the downscaled data to files
save(fullfile('tmp', 'train32x32.mat'), 'trainData32x32', 'trainLabels');
save(fullfile('tmp', 'validation32x32.mat'), 'valData32x32', 'valLabels');
save(fullfile('tmp', 'test32x32.mat'), 'testData32x32');

% Normalize every data points (5 points)
% Subtract the pixel mean
% Set the image norm (L2 vector norm on vectorized image matrix) to 10

%-------------------------------------------------------------------
%       Your code here (replace the code when necessary)

trainData32x32Normalized = zeros(nTrain, 32*32);
valData32x32Normalized = zeros(nVal, 32*32);
testData32x32Normalized = zeros(nTest, 32*32);
for i = 1 : nTrain
    avg = mean(trainData32x32(i, :));
    tempData = trainData32x32(i, :) - avg;
    normResult = norm(tempData);
    trainData32x32Normalized(i, :) = tempData/normResult*10;
end
for i = 1 : nVal
    avg = mean(valData32x32(i, :));
    tempData = valData32x32(i, :) - avg;
    normResult = norm(tempData);
    valData32x32Normalized(i, :) = tempData/normResult*10;
end
for i = 1 : nTest
    avg = mean(testData32x32(i, :));
    tempData = testData32x32(i, :) - avg;
    normResult = norm(tempData);
    testData32x32Normalized(i, :) = tempData/normResult*10;
end

%--------------------------------------------------------------------

% Save the normalized data to files
save(fullfile('tmp', 'train32x32Normalized.mat'), 'trainData32x32Normalized', 'trainLabels');
save(fullfile('tmp', 'validation32x32Normalized.mat'), 'valData32x32Normalized', 'valLabels');
save(fullfile('tmp', 'test32x32Normalized.mat'), 'testData32x32Normalized');

% Standardize data to zero mean and unit variance (5 points)
% Calculate the image mean (the mean of each pixel across all training data points)
% Useful Matlab function: mean
% Substract each training/validation data points by the image mean (pixel-wise)
% Calculate the standard deviation (the deviation of each pixel across all training data points)
% Useful Matlab function: std
% Divide each training/validation data points by the standard deviation (pixel-wise)

%-------------------------------------------------------------------
%       Your code here (replace the code when necessary)

trainData32x32NormalizedStandardized = zeros(nTrain, 32*32);
valData32x32NormalizedStandardized = zeros(nVal, 32*32);
testData32x32NormalizedStandardized = zeros(nTest, 32*32);
for i = 1 : 32*32
    meanVal = mean(trainData32x32Normalized(:, i));
    tempData = trainData32x32Normalized(:, i) - meanVal;
    stdDev = std(tempData);
    trainData32x32NormalizedStandardized(:, i) = tempData / stdDev;
    valData32x32NormalizedStandardized(:, i) = (valData32x32Normalized(:, i) - meanVal) / stdDev;
    testData32x32NormalizedStandardized(:, i) = (testData32x32Normalized(:, i) - meanVal) / stdDev;
end
%--------------------------------------------------------------------

% clean up variables that are no longer used
clear trainData32x32 trainData32x32Normalized valData32x32 valData32x32Normalized ...
    testData32x32 testData32x32Normalized;

% Save the standardized data to files
save(fullfile('tmp', 'train32x32NormalizedStandardized.mat'), 'trainData32x32NormalizedStandardized', 'trainLabels');
save(fullfile('tmp', 'validation32x32NormalizedStandardized.mat'), 'valData32x32NormalizedStandardized', 'valLabels');
save(fullfile('tmp', 'test32x32NormalizedStandardized.mat'), 'testData32x32NormalizedStandardized');

%% 2.b Classification with Convolutional neural network (15 points)

% We will use MatConvNet toolbox (http://www.vlfeat.org/matconvnet/) to model
% the network and perform training/predicting on the dataset.
%
% MatConvNet documentation
% - Function index: http://www.vlfeat.org/matconvnet/functions/
% - Manual: http://www.vlfeat.org/matconvnet/matconvnet-manual.pdf
% - FAQ: http://www.vlfeat.org/matconvnet/faq/
%
% Put the decompressed files to "toolbox0/matconvnet-1.0-beta16/"
%
% You may need to recompile the mex libraries if you are not working in
% Windows. Please follow the instructions on the official website for
% compilation.
%
% Please check the network architecture in the written problem description (PDF)
% First, specify the network architecture in struct "net", the following is
% an example of a simple convolutional neural network with two convolution
% and one pooling layer for binary classification:
%
% ------
%
% net.layers = {};
%
% % Suppose the grayscale images are 6x6. We add a convolutional layer with 4
% % 5x5 filters, a pad of 0 and a stride of 1
% % The layerwise activation dimensions are transformed from 6x6x1 (6x6 image
% % with 1 visual channel) to 2x2x4
% % The model parameters for this layer will be 5x5x1x4 (5x5 filters, 1
% % channel in the previous layer and 4 channels in the current layer)
%
% net.layers{end+1} = struct('type', 'conv', ...
%     'weights', {{initW*randn(5,5,1,4, 'single'), ...
%     initB*randn(1,4, 'single')}}, ...
%     'stride', 1, ...
%     'pad', 0);
%
% % Apply nonlinear activation function after convolution
% % Here we use ReLU (Rectified Linear Unit)
%
% net.layers{end+1} = struct('type', 'relu');
%
% % Add a 2x2 max pooling layer with a pad of 0 and a stride of 2
% % The layerwise activation dimensions are transformed from 2x2x4 to 1x1x4
%
% net.layers{end+1} = struct('type', 'pool', ...
%     'method', 'max', ...
%     'pool', [2, 2], ...
%     'stride', 2, ...
%     'pad', 0);
%
% % Add a convolutional layer with 2 1x1 filters, a pad of 0 and 1 stride of 1
% % The layerwise activation dimensions are transformed from 1x1x4 to 1x1x2
% % This special case (convolution with 1x1 filters) is equivalent to the
% % full layerwise connections in standard neural networks (from a 4x1 layer
% % to a 2x1 layer)
%
% net.layers{end+1} = struct('type', 'conv', ...
%     'weights', {{initW*randn(1,1,4,2, 'single'), ...
%     initB*randn(1,2, 'single')}}, ...
%     'stride', 1, ...
%     'pad', 0);
%
% % Finally, we add a Softmax Loss layer as the error function
% % No nonlinear activation function after the last convolution and before
% the softmax
%
% net.layers{end+1} = struct('type', 'softmaxloss');
%
% ------

% The initial value for weights and biases in convolutional layers
% In this mini project, we just specify them for you.
% You will need to find out the optimal parameters by yourself in
% real-world practice.
initW = 1e-2;
initB = 1e-1;

%-------------------------------------------------------------------
%       Your code here (replace the code when necessary)
initConvnetModelSmall.layers = {};
initConvnetModelSmall.layers{end + 1} = struct('type', 'conv', ...
    'weights', {{initW*randn(5,5,1,32, 'single'), ...
    initB*randn(1,32, 'single')}}, ...
    'stride', 1, ...
    'pad', 2);
initConvnetModelSmall.layers{end + 1} = struct('type', 'relu');
initConvnetModelSmall.layers{end + 1} = struct('type', 'pool', ...
    'method', 'max', ...
    'pool', [3, 3], ...
    'stride', 2, ...
    'pad', 1);
initConvnetModelSmall.layers{end + 1} = struct('type', 'conv', ...
    'weights', {{initW*randn(5,5,32,64, 'single'), ...
    initB*randn(1,64, 'single')}}, ...
    'stride', 1, ...
    'pad', 2);
initConvnetModelSmall.layers{end + 1} = struct('type', 'relu');
initConvnetModelSmall.layers{end + 1} = struct('type', 'pool', ...
    'method', 'max', ...
    'pool', [3, 3], ...
    'stride', 2, ...
    'pad', 1);
initConvnetModelSmall.layers{end + 1} = struct('type', 'conv', ...
    'weights', {{initW*randn(8,8,64,2048, 'single'), ...
    initB*randn(1,2048, 'single')}}, ...
    'stride', 1, ...
    'pad', 0);
initConvnetModelSmall.layers{end + 1} = struct('type', 'relu');
initConvnetModelSmall.layers{end + 1} = struct('type', 'conv', ...
    'weights', {{initW*randn(1,1,2048,7, 'single'), ...
    initB*randn(1,7, 'single')}}, ...
    'stride', 1, ...
    'pad', 0);
initConvnetModelSmall.layers{end+1} = struct('type', 'softmaxloss');
%-------------------------------------------------------------------

% Now perform training on the defined network
% A wrapper function cnnTrain has been provided ('toolbox0/cnn/cnnTrain.m')
% You can check the usage using the command "help cnnTrain"

% Some training options that you can use without changes
opts.continue = true;
opts.gpus = [];
opts.expDir = fullfile('tmp', 'convnetSmall');
if exist(opts.expDir, 'dir') ~= 7, mkdir(opts.expDir); end

% In this mini project, we just specify the parameters for you
% You will need to find out the optimal parameters by yourself in
% real-world practice.
opts.learningRate = 1e-2;
opts.batchSize = 128;
opts.numEpochs = 30;

%-------------------------------------------------------------------
%       Your code here (replace the code when necessary)
addpath('toolbox0/cnn');
addpath(genpath('toolbox0/matconvnet-1.0-beta16'));
addpath(genpath('toolbox0/vlfeat-0.9.20/toolbox/mex'));
addpath('toolbox0/matconvnet-1.0-beta16/examples');
addpath('toolbox0/vlfeat-0.9.20/toolbox/misc');
addpath('toolbox0/matconvnet-1.0-beta16/matlab');
%rmpath('toolbox0/matconvnet-1.0-beta16/matlab');
rmpath('toolbox0/vlfeat-0.9.20/toolbox/misc');
rmpath('toolbox0/matconvnet-1.0-beta16/matlab/vl_nnpool');
convnetModelSmall = cnnTrain(trainData32x32NormalizedStandardized,...
                            trainLabels, ...
                            valData32x32NormalizedStandardized,...
                            valLabels,...
                            initConvnetModelSmall,...
                            opts);
%-------------------------------------------------------------------

% Predict on the validation data
% A wrapper function cnnPredict has been provided ('toolbox0/cnn/cnnPredict.m')
convnetValPredictionSmall = cnnPredict(valData32x32NormalizedStandardized, convnetModelSmall, opts);
convnetValAccuracySmall = mean(valLabels == convnetValPredictionSmall)*100;
fprintf('Convolutional neural network (small) validation accuracy: %g%%\n', convnetValAccuracySmall);
convnetTestPredictionSmall = cnnPredict(testData32x32NormalizedStandardized, convnetModelSmall, opts);

% Save your networks and predictions to files
save(fullfile('tmp', 'convnetModelSmall.mat'), 'convnetModelSmall');
save(fullfile('result', 'convnetSmall.mat'), 'convnetValPredictionSmall', ...
    'convnetValAccuracySmall', 'convnetTestPredictionSmall');
% Get the iteration-error plot
copyfile(fullfile(opts.expDir, 'net-train.pdf'), fullfile('result', 'convnetPlotSmall.pdf'));


%% 3 Control experiments on convolutional neural networks (30 points)

% In this task, you will explore the perforance changes of convolutional
% neural networks when using different loss functions, activation functions
% and numbers of filters

% Refer to the code in task 2 for training and predicting
%
% MatConvNet documentation
% - Function index: http://www.vlfeat.org/matconvnet/functions/
% - Manual: http://www.vlfeat.org/matconvnet/matconvnet-manual.pdf
% - FAQ: http://www.vlfeat.org/matconvnet/faq/
%

% Base on the network architecture in task 2

% During training, the iteration-error plot will be automatically shown and
% saved to "net-train.pdf" in the folder opts.expDir (a subfolder in the
% temporary folder "tmp")
% Please copy "net-train.pdf" to the "result" folder and rename it
% accordingly

%% 3.a Try different loss functions (10 points)

% Change the loss function: L2 (squared) loss with softmax,
% multiclass structured hinge loss

%-------------------------------------------------------------------
%       Your code here (replace the code when necessary)

opts.expDir = fullfile('tmp', 'convnetSmallL2');
if exist(opts.expDir, 'dir') ~= 7, mkdir(opts.expDir); end
initConvnetModelSmallL2 = initConvnetModelSmall;
initConvnetModelSmallL2.layers{end} = struct('type', 'softmax');
initConvnetModelSmallL2.layers{end+1} = struct('type', 'pdist',...
                                        'p', 2,...
                                        'epsilon', 0,...
                                        'noRoot', true);
convnetModelSmallL2 = cnnTrain(trainData32x32NormalizedStandardized,...
                            trainLabels, ...
                            valData32x32NormalizedStandardized,...
                            valLabels,...
                            initConvnetModelSmallL2,...
                            opts);
convnetValPredictionSmallL2 = cnnPredict(valData32x32NormalizedStandardized, convnetModelSmallL2, opts);
convnetTestPredictionSmallL2 = cnnPredict(testData32x32NormalizedStandardized, convnetModelSmallL2, opts);
convnetValAccuracySmallL2 = mean(valLabels == convnetValPredictionSmallL2)*100;
fprintf('Convolutional neural network (small, L2) validation accuracy: %g%%\n', convnetValAccuracySmallL2);
copyfile(fullfile(opts.expDir, 'net-train.pdf'), fullfile('result', 'convnetPlotSmallL2.pdf'));


opts.expDir = fullfile('tmp', 'convnetSmallHinge');
if exist(opts.expDir, 'dir') ~= 7, mkdir(opts.expDir); end
initConvnetModelSmallHinge = initConvnetModelSmall;
initConvnetModelSmallHinge.layers{end} = struct('type','loss', ...
                                'opts', struct('loss', 'mshinge'));
convnetModelSmallHinge = cnnTrain(trainData32x32NormalizedStandardized,...
                            trainLabels, ...
                            valData32x32NormalizedStandardized,...
                            valLabels,...
                            initConvnetModelSmallHinge,...
                            opts);
convnetValPredictionSmallHinge = cnnPredict(valData32x32NormalizedStandardized, convnetModelSmallHinge, opts);
convnetTestPredictionSmallHinge = cnnPredict(testData32x32NormalizedStandardized, convnetModelSmallHinge, opts);
convnetValAccuracySmallHinge = mean(valLabels == convnetValPredictionSmallHinge)*100;
fprintf('Convolutional neural network (small, hinge) validation accuracy: %g%%\n', convnetValAccuracySmallHinge);
copyfile(fullfile(opts.expDir, 'net-train.pdf'), fullfile('result', 'convnetPlotSmallHinge.pdf'));

%-------------------------------------------------------------------

% Save your results to files
save(fullfile('tmp', 'convnetModelSmallTryLoss.mat'), 'convnetModelSmallL2', 'convnetModelSmallHinge');
save(fullfile('result', 'convnetSmallTryLoss.mat'), 'convnetValPredictionSmallL2', ... 
    'convnetTestPredictionSmallL2', 'convnetValAccuracySmallL2', 'convnetValPredictionSmallHinge', ...
    'convnetTestPredictionSmallHinge', 'convnetValAccuracySmallHinge');


%% 3.b Try different activation functions (10 points)

% Change the activation functions: no activation functions, sigmoid
% Don't mix the activation functions in the same network

%-------------------------------------------------------------------
%       Your code here (replace the code when necessary)

opts.expDir = fullfile('tmp', 'convnetSmallNoActivationFunc');
if exist(opts.expDir, 'dir') ~= 7, mkdir(opts.expDir); end
initConvnetModelSmallNoActivationFunc = initConvnetModelSmall;
initConvnetModelSmallNoActivationFunc.layers{2} = [];
initConvnetModelSmallNoActivationFunc.layers{5} = [];
initConvnetModelSmallNoActivationFunc.layers{8} = [];
% remove empty cells
initConvnetModelSmallNoActivationFunc.layers = ...
                        initConvnetModelSmallNoActivationFunc.layers...
                        (~cellfun('isempty',initConvnetModelSmallNoActivationFunc.layers)); 
convnetModelSmallNoActivationFunc = cnnTrain(trainData32x32NormalizedStandardized,...
                            trainLabels, ...
                            valData32x32NormalizedStandardized,...
                            valLabels,...
                            initConvnetModelSmallNoActivationFunc,...
                            opts);
convnetValPredictionSmallNoActivationFunc = cnnPredict(valData32x32NormalizedStandardized, convnetModelSmallNoActivationFunc, opts);
convnetTestPredictionSmallNoActivationFunc = cnnPredict(testData32x32NormalizedStandardized, convnetModelSmallNoActivationFunc, opts);
convnetValAccuracySmallNoActivationFunc = mean(valLabels == convnetValPredictionSmallNoActivationFunc)*100;
fprintf('Convolutional neural network (small, no activation func.) accuracy: %g%%\n', convnetValAccuracySmallNoActivationFunc);
copyfile(fullfile(opts.expDir, 'net-train.pdf'), fullfile('result', 'convnetPlotSmallNoActivationFunc.pdf'));

opts.expDir = fullfile('tmp', 'convnetSmallSigmoid');
if exist(opts.expDir, 'dir') ~= 7, mkdir(opts.expDir); end
initConvnetModelSmallSigmoid = initConvnetModelSmall;
initConvnetModelSmallSigmoid.layers{2} = struct('type', 'sigmoid');
initConvnetModelSmallSigmoid.layers{5} = struct('type', 'sigmoid');
initConvnetModelSmallSigmoid.layers{8} = struct('type', 'sigmoid');
convnetModelSmallSigmoid = cnnTrain(trainData32x32NormalizedStandardized,...
                            trainLabels, ...
                            valData32x32NormalizedStandardized,...
                            valLabels,...
                            initConvnetModelSmallSigmoid,...
                            opts);
convnetValPredictionSmallSigmoid = cnnPredict(valData32x32NormalizedStandardized, convnetModelSmallSigmoid, opts);
convnetTestPredictionSmallSigmoid = cnnPredict(testData32x32NormalizedStandardized, convnetModelSmallSigmoid, opts);
convnetValAccuracySmallSigmoid = mean(valLabels == convnetValPredictionSmallSigmoid)*100;
fprintf('Convolutional neural network (small, sigmoid) accuracy: %g%%\n', convnetValAccuracySmallSigmoid);
copyfile(fullfile(opts.expDir, 'net-train.pdf'), fullfile('result', 'convnetPlotSmallSigmoid.pdf'));

%-------------------------------------------------------------------

% Save your results to files
save(fullfile('tmp', 'convnetModelSmallTryActivationFunc.mat'), ...
    'convnetModelSmallNoActivationFunc', 'convnetModelSmallSigmoid');
save(fullfile('result', 'convnetSmallTryActivationFunc.mat'), ...
    'convnetValPredictionSmallNoActivationFunc', 'convnetValAccuracySmallNoActivationFunc', ...
    'convnetTestPredictionSmallNoActivationFunc', 'convnetValPredictionSmallSigmoid', ...
    'convnetValAccuracySmallSigmoid', 'convnetTestPredictionSmallSigmoid');


%% 3.c Try different numbers of filters (10 points)
%
% Change the numbers of filters in the convolutional layers (note: the pooling
% layer can't change the number of filters)
% Scale number at the same time for all convolutional layers
% e.g. the 1st, 2nd and 3rd convolutional layers have 32, 64 and 2048 filters
%      -scaled by 2.0-> 64, 128, 4096 filters, respectively
%   or -scaled by 0.5-> 16, 32, 1024 filters, respectively
%
% Note that the number of filters in the pooling/ReLu layer will change
% correspondingly

% Try these scales
scale = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0];

convnetModelSmallFilterNumScaled = cell(numel(scale), 1);
convnetValPredictionSmallFilterNumScaled = cell(numel(scale), 1);
convnetTestPredictionSmallFilterNumScaled = cell(numel(scale), 1);
convnetValAccuracySmallFilterNumScaled = cell(numel(scale), 1);

%-------------------------------------------------------------------
%       Your code here (replace the code when necessary)

initConvnetModelSmallFilterNumScaled = initConvnetModelSmall;
for i = 1:numel(scale)
    
    opts.expDir = fullfile('tmp', sprintf('convnetSmallFilterNumScaled_%s', num2str(scale(i))));
    if exist(opts.expDir, 'dir') ~= 7, mkdir(opts.expDir); end
    
    % Resize the filters
    initConvnetModelSmallFilterNumScaled.layers{1} = struct('type', 'conv', ...
                        'weights', {{initW*randn(5,5,1,scale(i)*32, 'single'), ...
                        initB*randn(1,scale(i)*32, 'single')}}, ...
                        'stride', 1, ...
                        'pad', 2);
    initConvnetModelSmallFilterNumScaled.layers{4} = struct('type', 'conv', ...
                        'weights', {{initW*randn(5,5,scale(i)*32,scale(i)*64, 'single'), ...
                        initB*randn(1,scale(i)*64, 'single')}}, ...
                        'stride', 1, ...
                        'pad', 2);
    initConvnetModelSmallFilterNumScaled.layers{7} = struct('type', 'conv', ...
                        'weights', {{initW*randn(8,8,scale(i)*64,scale(i)*2048, 'single'), ...
                        initB*randn(1,scale(i)*2048, 'single')}}, ...
                        'stride', 1, ...
                        'pad', 0);
    initConvnetModelSmallFilterNumScaled.layers{9} = struct('type', 'conv', ...
                        'weights', {{initW*randn(1,1,scale(i)*2048,7, 'single'), ...
                        initB*randn(1,7, 'single')}}, ...
                        'stride', 1, ...
                        'pad', 0);
    convnetModelSmallFilterNumScaled{i} = cnnTrain(trainData32x32NormalizedStandardized,...
                            trainLabels, ...
                            valData32x32NormalizedStandardized,...
                            valLabels,...
                            initConvnetModelSmallFilterNumScaled,...
                            opts);
    convnetValPredictionSmallFilterNumScaled{i} = cnnPredict(valData32x32NormalizedStandardized, convnetModelSmallFilterNumScaled{i}, opts);
    convnetTestPredictionSmallFilterNumScaled{i} = cnnPredict(testData32x32NormalizedStandardized, convnetModelSmallFilterNumScaled{i}, opts);
    convnetValAccuracySmallFilterNumScaled{i} = mean(valLabels == convnetValPredictionSmallFilterNumScaled{i})*100;
    
    fprintf('Convolutional neural network (small, filter # scaled by: %s) validation accuracy: %g%%\n', ...
        num2str(scale(i)), convnetValAccuracySmallFilterNumScaled{i});
    copyfile(fullfile(opts.expDir, 'net-train.pdf'), fullfile('result', 'convnetPlotSmallFilterNumScaled.pdf'));
end

%-------------------------------------------------------------------

% Save your results to files
save(fullfile('tmp', 'convnetModelSmallFilterNumScaled.mat'), 'convnetModelSmallFilterNumScaled');
save(fullfile('result', 'convnetSmallFilterNumScaled.mat'), 'convnetValPredictionSmallFilterNumScaled', ...
    'convnetTestPredictionSmallFilterNumScaled', 'convnetValAccuracySmallFilterNumScaled');


%% 4 Better convolutional neural networks (20 points + 10 extra credit points)

% clean up variables that are no longer used
clear trainData32x32NormalizedStandardized valData32x32NormalizedStandardized ...
    testData32x32NormalizedStandardized;

%% 4.a A larger convolutional neural networks (20 points)

% Now we will train a larger convolutional neural network on the full
% version (48x48) data
%
% Please check the network architecture in the written problem description (PDF) 
% 
% MatConvNet documentation
% - Function index: http://www.vlfeat.org/matconvnet/functions/
% - Manual: http://www.vlfeat.org/matconvnet/matconvnet-manual.pdf
% - FAQ: http://www.vlfeat.org/matconvnet/faq/
%

% Normalize and standardize the 48x48 images (5 points)
%-------------------------------------------------------------------
%       Your code here (replace the code when necessary)

trainData48x48NormalizedStandardized = [];
valData48x48NormalizedStandardized = [];
testData48x48NormalizedStandardized = [];
for i = 1 : nTrain
    avg = mean(trainData(i, :));
    tempData = trainData(i, :) - avg;
    normResult = norm(tempData);
    trainData48x48NormalizedStandardized(i, :) = tempData/normResult*10;
end
for i = 1 : nVal
    avg = mean(valData(i, :));
    tempData = valData(i, :) - avg;
    normResult = norm(tempData);
    valData48x48NormalizedStandardized(i, :) = tempData/normResult*10;
end
for i = 1 : nTest
    avg = mean(testData(i, :));
    tempData = testData(i, :) - avg;
    normResult = norm(tempData);
    testData48x48NormalizedStandardized(i, :) = tempData/normResult*10;
end
for i = 1 : 48*48
    meanVal = mean(trainData48x48NormalizedStandardized(:, i));
    tempData = trainData48x48NormalizedStandardized(:, i) - meanVal;
    stdDev = std(tempData);
    trainData48x48NormalizedStandardized(:, i) = tempData / stdDev;
    valData48x48NormalizedStandardized(:, i) = (valData48x48NormalizedStandardized(:, i) - meanVal) / stdDev;
    testData48x48NormalizedStandardized(:, i) = (testData48x48NormalizedStandardized(:, i) - meanVal) / stdDev;
end
%-------------------------------------------------------------------
%%
opts.expDir = fullfile('tmp', 'convnetLarge');
if exist(opts.expDir, 'dir') ~= 7, mkdir(opts.expDir); end

% Train the model (15 points)
%-------------------------------------------------------------------
%       Your code here (replace the code when necessary)
initConvnetModelLarge.layers = {};
initConvnetModelLarge.layers{end + 1} = struct('type', 'conv', ...
    'weights', {{initW*randn(5,5,1,32, 'single'), ...
    initB*randn(1,32, 'single')}}, ...
    'stride', 1, ...
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
    'weights', {{initW*randn(1,1,4096,7, 'single'), ...
    initB*randn(1,7, 'single')}}, ...
    'stride', 1, ...
    'pad', 0);
initConvnetModelLarge.layers{end+1} = struct('type', 'softmaxloss');
convnetModelLarge = cnnTrain(trainData48x48NormalizedStandardized,...
                            trainLabels, ...
                            valData48x48NormalizedStandardized,...
                            valLabels,...
                            initConvnetModelLarge,...
                            opts);

%-------------------------------------------------------------------

% Predict on the validation data
convnetValPredictionLarge = cnnPredict(valData48x48NormalizedStandardized, convnetModelLarge, opts);
convnetTestPredictionLarge = cnnPredict(testData48x48NormalizedStandardized, convnetModelLarge, opts);
convnetValAccuracyLarge = mean(valLabels == convnetValPredictionLarge)*100;
fprintf('Convolutional neural network (large) validation accuracy: %g%%\n', convnetValAccuracyLarge);

% Save your networks and predictions to files
save(fullfile('tmp', 'convnetModelLarge.mat'), 'convnetModelLarge');
save(fullfile('result', 'convnetLarge.mat'), 'convnetValPredictionLarge', ...
    'convnetTestPredictionLarge', 'convnetValAccuracyLarge');
% Get the iteration-error plot
copyfile(fullfile(opts.expDir, 'net-train.pdf'), fullfile('result', 'convnetPlotLarge.pdf'));


%% 4.b Explore your convolutional neural networks  (20 extra credit points)

% Extra credit points are available for convolutional neural networks that
% outperform the convolutional neural networks in previous task.
% You may utilize your findings in task 3 to improve your convolutional neural networks

opts.expDir = fullfile('tmp', 'convnetExtra');
if exist(opts.expDir, 'dir') ~= 7, mkdir(opts.expDir); end

% Train the model
%-------------------------------------------------------------------
%       Your code here (replace the code when necessary)

convnetModelExtra = [];

%-------------------------------------------------------------------

% Predict on the validation data
convnetValPredictionExtra = cnnPredict(valData48x48NormalizedStandardized, convnetModelExtra, opts);
convnetTestPredictionExtra = cnnPredict(testData48x48NormalizedStandardized, convnetModelExtra, opts);
convnetValAccuracyExtra = mean(valLabels == convnetValPredictionExtra')*100;
printf('Convolutional neural network (large) accuracy: %g%%\n', convnetValAccuracyExtra);

% Save your networks and predictions to files
save(fullfile('tmp', 'convnetModelExtra.mat'), 'convnetModelExtra');
save(fullfile('result', 'convnetExtra.mat'), 'convnetValPredictionExtra', ...
    'convnetTestPredictionExtra', 'convnetValAccuracyExtra');
% Get the iteration-error plot
copyfile(fullfile(opts.expDir, 'net-train.pdf'), fullfile('result', 'convnetPlotExtra.pdf'));

%% Submission

% Please provide your unique name and the list of files to pack up

%-------------------------------------------------------------------
%       Your code here (replace the code when necessary)
uniqname = snofz;
filelist = {'result/', 'toolbox0/', 'main.m'};
%-------------------------------------------------------------------

zip(sprintf('fer_%s.zip', uniqname), filelist);
% pack all the required files to a single zip file
% submit this file to CTools

