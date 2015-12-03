function predictions = cnnPredict(testData, model, opts)
% predictions = cnnPredict(testData, model, opts)
% Convolutional neural network predicting wrapper for MatConvNet, return
% predictions, the predictions on the test data
%
% Arguments:
% testData - N_{test}xM matrix, where each row is a test data point
% model - the learned convolutional neural network model
% opts - options
%   opts.batchSize - the size of batches for forward propagation
%   opts.numEpochs - the number of iterations in gradient descent
%   opts.learningRate - the learning rate for gradient descent
%

for i = 1:numel(model.layers)-1, predictNet.layers{i} = model.layers{i}; end

nTest = size(testData, 1);
imHeight = sqrt(size(testData, 2));
imWidth = imHeight;
data = single(reshape(testData', imHeight, imWidth, 1, nTest));
predictions = zeros(nTest, 1);

for t=1:opts.batchSize:nTest
    batch = t:min(t+opts.batchSize-1, nTest);
    fprintf('predicting: batch %d/%d ...\n', fix(t/opts.batchSize)+1, ceil(nTest/opts.batchSize));
    res = vl_simplenn(predictNet, data(:, :, :, batch), [], [], 'disableDropout', true);
    [~, predictedLabels] = max(res(end).x, [], 3);
    predictions(batch) = squeeze(predictedLabels);
end


end
