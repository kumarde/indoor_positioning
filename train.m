% Carisa Covins
% Alan Lundgard
% Deepak Kumar
% Spencer Nofzinger
% Sudhanva Sreesha
% EECS 445 - Project
% Train

function svm_structs = train(train_matrix, train_labels)
	unique_labels = unique(train_labels);
	for i = 1:length(unique_labels)
		classes = (train_labels == unique_labels(i));
		svm_structs(i) = svmtrain(train_matrix, classes);
	end
end