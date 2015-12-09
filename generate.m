% Carisa Covins
% Alan Lundgard
% Deepak Kumar
% Spencer Nofzinger
% Sudhanva Sreesha
% EECS 445 - Project
% Generate Data Matrices

function [matrix, labels] = generate(set, type)
	cache = ['cache/data_' set '_' type '_sift.mat'];
	if exist(cache) == 2
		load(cache);
		return
	end

	matrix = {};
	labels = [];

	data_folder = ['data_' set '/' type];
	sub_directories = dir(data_folder);

	filters = ismember({sub_directories.name}, {'.', '..'});
	sub_directories(filters) = [];
	image_count = 1;

	for i = 1:length(sub_directories)
		current_directory = sub_directories(i).name;
		directory_path = [data_folder '/' current_directory];

		images = dir(directory_path);
		filters = ismember({images.name}, {'.', '..'});
		images(filters) = [];

		for j = 1:length(images)
			fprintf('Dealing with image %s_%d.tiff\n', current_directory, j);
			file_name = [directory_path '/' images(j).name];
			[points, features] = prepare(file_name, 'SIFT');
			matrix{image_count} = features;
			image_count = image_count + 1;
			labels = [labels; i];
		end
	end

	save(cache, 'matrix', 'labels');
end
