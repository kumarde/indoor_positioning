
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

% Some training options that you can use without changes
opts.continue = false;
opts.gpus = [];

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
rmpath('toolbox0/vlfeat-0.9.20/toolbox/misc');
rmpath('toolbox0/matconvnet-1.0-beta16/matlab/vl_nnpool');
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

load('tmp/train.mat');
load('tmp/val.mat');

%
[nTrain, ~] = size(train_data);
[nVal, ~] = size(val_data);
trainData = zeros(nTrain, 96*96*3);
valData = zeros(nVal, 96*96*3);
for i = 1 : nTrain
    trainData(i, :) = reshape(imresize(reshape(train_data(i, :), 612, 816, 3),[96, 96]), 1, 96*96*3); 
end
for i = 1 : nVal
    valData(i, :) = reshape(imresize(reshape(val_data(i, :), 612, 816, 3),[96,96]), 1, 96*96*3); 
end
trainData = double(trainData);
valData = double(valData);
trainLabels = double(train_labels);
valLabels = double(val_labels);
%
% Grayscale
%{
[nTrain, ~] = size(train_data);
[nVal, ~] = size(val_data);
trainData = zeros(nTrain, 96*96);
valData = zeros(nVal, 96*96);
for i = 1 : nTrain
    trainData(i, :) = reshape(imresize(rgb2gray(reshape(train_data(i, :), 612, 816, 3)), [96, 96]), 1, 96*96); 
end
for i = 1 : nVal
    valData(i, :) = reshape(imresize(rgb2gray(reshape(val_data(i, :), 612, 816, 3)),[96, 96]), 1, 96*96); 
end
trainData = double(trainData);
valData = double(valData);
trainLabels = double(train_labels);
valLabels = double(val_labels);
%}
%
%%
% RGB attempt
%{
[nTrain, ~] = size(train_data);
[nVal, ~] = size(val_data);
trainData = zeros(nTrain, 96*96, 3);
valData = zeros(nVal, 96*96, 3);
for i = 1 : nTrain
    trainData(i, :, :) = reshape(imresize(reshape(train_data(i, :), 612, 816, 3),[96, 96]), 1, 96*96, 3); 
end
for i = 1 : nVal
    valData(i, :, :) = reshape(imresize(reshape(val_data(i, :), 612, 816, 3),[96,96]), 1, 96*96, 3); 
end
trainData = double(trainData);
valData = double(valData);
trainLabels = double(train_labels);
valLabels = double(val_labels);
%}


%
train_data_normalized_standardized = [];
val_data_normalized_standardized = [];
[~, imLength] = size(trainData);
for i = 1 : nTrain
    avg = mean(trainData(i, :));
    tempData = trainData(i, :) - avg;
    normResult = norm(tempData);
    train_data_normalized_standardized(i, :) = tempData/normResult*10;
end
for i = 1 : nVal
    avg = mean(valData(i, :));
    tempData = valData(i, :) - avg;
    normResult = norm(tempData);
    val_data_normalized_standardized(i, :) = tempData/normResult*10;
end
for i = 1 : imLength
    meanVal = mean(train_data_normalized_standardized(:, i));
    tempData = train_data_normalized_standardized(:, i) - meanVal;
    stdDev = std(tempData);
    train_data_normalized_standardized(:, i) = tempData / stdDev;
    val_data_normalized_standardized(:, i) = (val_data_normalized_standardized(:, i) - meanVal) / stdDev;
end
%

%RGB 3 DIM
%{
train_data_normalized_standardized = [];
val_data_normalized_standardized = [];
[~, imLength] = size(trainData);
for i = 1 : nTrain
    for dim = 1 : 3
        avg = mean(trainData(i, :, dim));
        tempData = trainData(i, :, dim) - avg;
        normResult = norm(tempData);
        train_data_normalized_standardized(i, :, dim) = tempData/normResult*10;
    end
end
for i = 1 : nVal
    for dim = 1:3
        avg = mean(valData(i, :, dim));
        tempData = valData(i, :, dim) - avg;
        normResult = norm(tempData);
        val_data_normalized_standardized(i, :, dim) = tempData/normResult*10;
    end
end
for i = 1 : 96*96
    for dim = 1:3
        meanVal = mean(train_data_normalized_standardized(:, i, dim));
        tempData = train_data_normalized_standardized(:, i, dim) - meanVal;
        stdDev = std(tempData);
        train_data_normalized_standardized(:, i, dim) = tempData / stdDev;
        val_data_normalized_standardized(:, i, dim) = (val_data_normalized_standardized(:, i, dim) - meanVal) / stdDev;
    end
end
%}

%%
%{
red = zeros(nTrain, 48*48);
green = red;
blue = red;
for i = 1 : nTrain
    temp = reshape(train_data_normalized_standardized(i, :), 48, 48, 3);
    red(i, :) = reshape(temp(:, : ,1), 1, 48*48);
    green(i, :) = reshape(temp(:, : ,2), 1, 48*48);
    blue(i, :) = reshape(temp(:, : ,3), 1, 48*48);
end

red2 = zeros(nVal, 48*48);
green2 = red2;
blue2 = red2;
for i = 1 : nVal
    temp = reshape(val_data_normalized_standardized(i, :), 48, 48, 3);
    red2(i, :) = reshape(temp(:, : ,1), 1, 48*48);
    green2(i, :) = reshape(temp(:, : ,2), 1, 48*48);
    blue2(i, :) = reshape(temp(:, : ,3), 1, 48*48);
end
%}

%-------------------------------------------------------------------
%%
opts.expDir = fullfile('tmp', 'convnetLarge');
if exist(opts.expDir, 'dir') ~= 7, mkdir(opts.expDir); end

% Train the model (15 points)
%-------------------------------------------------------------------
%       Your code here (replace the code when necessary)
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
    'weights', {{initW*randn(1,1,4096,5, 'single'), ...
    initB*randn(1,5, 'single')}}, ...
    'stride', 1, ...
    'pad', 0);
initConvnetModelLarge.layers{end+1} = struct('type', 'softmaxloss');


convnetModelLarge = cnnTrain(train_data_normalized_standardized,...
                            trainLabels, ...
                            val_data_normalized_standardized,...
                            valLabels,...
                            initConvnetModelLarge,...
                            opts);

%-------------------------------------------------------------------

% Predict on the validation data
convnetValPredictionLarge = cnnPredict(val_data_normalized_standardized, convnetModelLarge, opts);
convnetValAccuracyLarge = mean(valLabels == convnetValPredictionLarge)*100;
fprintf('Convolutional neural network (large) validation accuracy: %g%%\n', convnetValAccuracyLarge);

% Save your networks and predictions to files
save(fullfile('tmp', 'convnetModelLarge.mat'), 'convnetModelLarge');
save(fullfile('result', 'convnetLarge.mat'), 'convnetValPredictionLarge', ...
     'convnetValAccuracyLarge');
% Get the iteration-error plot
copyfile(fullfile(opts.expDir, 'net-train.pdf'), fullfile('result', 'convnetPlotLarge.pdf'));


