% Carisa Covins
% Alan Lundgard
% Deepak Kumar
% Spencer Nofzinger
% Sudhanva Sreesha
% EECS 445 - Project

% A function that loads a reference image (specified in the input
% agrument 'image') and obtains a feature space with a moethod
% indicated by the input argument 'method'. The method options
% for the feature detector include: SURF, SIFT, Harris Corners.
% The function ouputs the time taken to perfrom the feature
% detection and the keypoints as described by each of the
% algorithms.

function [referencePoints, referenceFeatures] = prepare(name, method)
	% Initliaze Output Arguments
	referencePoints   = [];
	referenceFeatures = [];

	grayscale = rgb2gray(imread(name));
	if strcmp(method, 'SURF') == 1
		referencePoints = detectSURFFeatures(grayscale);
		referenceFeatures = extractFeatures(grayscale, referencePoints);
	elseif strcmp(method, 'SIFT') == 1
		[referecePoints, referenceFeatures] = vl_sift(single(grayscale));
    end
end
