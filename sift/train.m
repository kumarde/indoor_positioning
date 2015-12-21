% Carisa Covins
% Alan Lundgard
% Deepak Kumar
% Spencer Nofzinger
% Sudhanva Sreesha
% EECS 445 - Project
% Train

function models = train(train_matrix)
	for i = 1:length(train_matrix)
		models{i} = vl_kdtreebuild(single(train_matrix{i}));
	end
end
