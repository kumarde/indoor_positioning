% Carisa Covins
% Alan Lundgard
% Deepak Kumar
% Spencer Nofzinger
% Sudhanva Sreesha
% EECS 445 - Project
% Generate Historgram Plot

% data_one/test/hall/20151107_141134.jpg
function historgram_plot(file)
% Read in standard MATLAB color demo image.
	image = imread(file);
	[height wqidth dimensions] = size(image);

	subplot(2, 2, 1);
	imshow(image, []);
	title('Original Image');
	set(gcf, 'Position', get(0,'Screensize'));

	red_image = image(:, :, 1);
	green_image = image(:, :, 2);
	blue_image = image(:, :, 3);

	[pixels levels] = imhist(red_image);
	subplot(2, 2, 2);
	plot(pixels, 'r');
	title('Red Histogram');
	xlim([0 levels(end)]);

	[pixels levels] = imhist(green_image);
	subplot(2, 2, 3);
	plot(pixels, 'g');
	title('Green Histogram');
	xlim([0 levels(end)]);

	[pixels levels] = imhist(blue_image);
	subplot(2, 2, 4);
	plot(pixels, 'b');
	title('Blue Histogram');
	xlim([0 levels(end)]);
end