% Carisa Covins
% Alan Lundgard
% Deepak Kumar
% Spencer Nofzinger
% Sudhanva Sreesha
% EECS 445 - Project
% Test

function [labels, accuracy] = test(test_matirx, test_labels, svm_models)
	[N, M] = size(test_matirx);
	labels = zeros(N, 1);

	for i = 1:N
		for j = 1:length(svm_models)
			if svmclassify(svm_models(j), test_matirx(i, :))
				break
			end
		end
		labels(i) = j;
	end

	correct = find(labels == test_labels);
	accuracy = (length(correct) / length(test_labels)) * 100;
end