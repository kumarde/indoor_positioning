function net = cnnTrainGray(trainData, trainLabels, valData, valLabels, initNet, opts)
% Same as cnnTrain from miniproject, 1 dimension instead of 3 for RGB


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
