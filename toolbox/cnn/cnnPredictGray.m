function predictions = cnnPredict(testData, model, opts)
% Same as cnnPredict from mini project, no 3rd dimension for RGB

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
