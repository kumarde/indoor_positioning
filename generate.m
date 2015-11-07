% Carisa Covins
% Alan Lundgard
% Deepak Kumar
% Spencer Nofzinger
% Sudhanva Sreesha
% EECS 445 - Project
% Generate Data Matrices

function [matrix, labels] = generate(type)
	matrix = [];
	labels = [];

	data_folder = ['data/' type];
	sub_directories = dir(data_folder);

	filters = ismember({sub_directories.name}, {'.', '..'});
	sub_directories(filters) = [];

	for i = 1:length(sub_directories)
		current_directory = sub_directories(i).name;
		directory_path = [data_folder '/' current_directory];

		images = dir(directory_path);
		filters = ismember({images.name}, {'.', '..'});
		images(filters) = [];

		for j = 1:length(images)
			file_name = [directory_path '/' images(j).name];
			H = histogram(file_name, 10, 2);
			matrix = [matrix; H'];
			labels = [labels; i];
		end
	end
end