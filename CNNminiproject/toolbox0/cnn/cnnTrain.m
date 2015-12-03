function net = cnnTrain(trainData, trainLabels, valData, valLabels, initNet, opts)
% net = cnnTrain(trainData, trainLabels, valData, valLabels, initNet, opts)
% Convolutional neural network training wrapper for MatConvNet, return net,
% the trained model
%
% Arguments:
% trainData - N_{train}xM matrix, where each row is a training data point
% trainLabels - N_{train}x1 vector, where each row is the label for a training
% data point
% valData - N_{validation}xM matrix, where each row is a training data point
% valLabels - N_{validation}x1 vector, where each row is the label for a
% validation data point
% initNet - the network architecture defined with initialized weights and
% bias
% opts - options
%   opts.learningRate - the learning rate for gradient descent
%   opts.batchSize - the size of batches for gradient descent
%   opts.numEpochs - the number of epochs(iterations) for gradient descent
%   opts.continue - a boolean indicating if we resume from the finished epochs
%   opts.gpus - a vector including the indexes of GPU devices to be used
%               (leave blank if using CPU)
%   opts.expDir - the directory for saving the network for each epoch (so
%                 that we can resume from a finished epochs after pause)
% 

nTrain = size(trainLabels, 1);
nVal = size(valLabels, 1);
imdb.imHeight = sqrt(size(trainData, 2));
imdb.imWidth = imdb.imHeight;
imdb.images.data = single(reshape(cat(1, trainData, valData)', imdb.imHeight, imdb.imWidth, 1, nTrain+nVal));
imdb.images.labels = single(cat(1, trainLabels, valLabels)');
imdb.images.set = cat(2, ones(1, nTrain), 3*ones(1, nVal));
[net, ~] = cnn_train(initNet, imdb, @getBatch, opts, 'val', find(imdb.images.set == 3));
net = vl_simplenn_move(net, 'cpu');

end

function [im, labels] = getBatch(imdb, batch)
im = imdb.images.data(:,:,:,batch);
labels = imdb.images.labels(1,batch);
end
