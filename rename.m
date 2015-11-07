% Carisa Covins
% Alan Lundgard
% Deepak Kumar
% Spencer Nofzinger
% Sudhanva Sreesha
% EECS 445 - Project
% Rename Images

function rename(type)
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
			old_name = [directory_path '/' images(j).name];
			new_name = [directory_path '/' current_directory];
			new_name = sprintf('%s_%d.jpg', new_name, j);
			if (~strcmp(old_name, new_name))
				movefile(old_name, new_name)
			end
		end
	end
end