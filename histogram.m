% Carisa Covins
% Alan Lundgard
% Deepak Kumar
% Spencer Nofzinger
% Sudhanva Sreesha
% EECS 445 - Project
% Generate Color Histogram

function H = histogram(image, bins, normalization)
	image = imread(image);
	[height, width, dimensions] = size(image);

	if dimensions ~= 3
		error ('Number of Dimensions in Image', 'Input image should only be an RGB image.');
	end

	if nargin < 3
		normalization = 0;
	end

	red_histogram		=	imhist(image(:, :, 1), bins);
	green_histogram	=	imhist(image(:, :, 2), bins);
	blue_histogram	=	imhist(image(:, :, 3), bins);

	H = [red_histogram green_histogram blue_histogram];

	% Un-normalized histogram
	H = H(:);
	% L1-Normalization
	if normalization == 1
		H = H ./ sum(H);
	% L2-Normalization
	else if normalization == 2
		H = normc(H);
	end
end
