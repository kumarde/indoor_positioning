% This creates the train data from the room photos. The path must point to
% a folder containing folders for each room, such as \bbb or \dow, with
% photos in each of those rooms. This code is commented out because it was
% modified and had to be broken up into different .mat files for each room
% because of the huge amount of North Campus Data Set images.

%clear;
%{
photos = 'C:\Users\Spencer\Documents\GitHub\indoor_positioning\indoor_positioning\temporary';
folders = dir(photos);
roomNames = {folders([folders.isdir]).name};
roomNames = roomNames(3:end);
numRooms = length(roomNames);
train_data = [];
train_labels = [];

imnum = 0;
for room = 1 : numRooms
    subfolder = strcat(strcat(photos,'\'), roomNames(room));
    subfolder = subfolder{1};
    subfolder = strcat(subfolder, '\');
    subfolderpath = strcat(subfolder, '*');
    images = dir(subfolderpath);
    images = images(3 : end);
    for j = 1 : numel(images(:, 1));
       train_labels = [train_labels; room]; 
       name = images(j).name;
       name = strcat(subfolder, name);
       image = imread(name);
       image = imresize(image, [612, 816]);
       image = reshape(image, [1, 612*816*3]);
       train_data = [train_data; image];
    end
end
%}
%%
% This creates the validation data, must be the same format as the train
% data folder, but in test folder instead.
photos = 'C:\Users\Spencer\Documents\GitHub\indoor_positioning\indoor_positioning\data_three\test';
folders = dir(photos);
roomNames = {folders([folders.isdir]).name};
roomNames = roomNames(3:end);
numRooms = length(roomNames);

val_data = [];
val_labels = [];

imnum = 0;
for room = 1 : numRooms
    subfolder = strcat(strcat(photos,'\'), roomNames(room));
    subfolder = subfolder{1};
    subfolder = strcat(subfolder, '\');
    subfolderpath = strcat(subfolder, '*');
    images = dir(subfolderpath);
    images = images(3 : end);
    for j = 1 : numel(images(:, 1));
       val_labels = [val_labels; room]; 
       name = images(j).name;
       name = strcat(subfolder, name);
       image = imread(name);
       image = imresize(image, [612, 816]);
       image = reshape(image, [1, 612*816*3]);
       val_data = [val_data; image];
    end
end

if ~exist('tmp', 'dir'), mkdir('tmp'); end
%save(fullfile('tmp', 'train.mat'), 'train_data', 'train_labels');
save(fullfile('tmp', 'val.mat'), 'val_data', 'val_labels');