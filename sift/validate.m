% Carisa Covins
% Alan Lundgard
% Deepak Kumar
% Spencer Nofzinger
% Sudhanva Sreesha
% EECS 445 - Project
% Test

function [labels, accuracy] = test( ...
		train_matrix,               ...
		train_labels,               ...
		test_matrix,                ...
		test_labels,                ...
		models)
	N = length(test_matrix);
	labels = zeros(N, 1);

	% Higher value allows for more matches.
	threshold = .47;

	for i = 1:N
		fprintf('Testing image %d\n', i);

		label = 1;
		maximum = intmin;

		for j = 1:length(models)
			fprintf('\tComparing with training image %d\n', j);

			index_pairs = [];

			% Approximates k Nearest Neighbors (with
			% k = 2) for each column of videoFeatures;
			% finds the two closest neighbors (columns
			% of base Features).
			[indeces, distances] = vl_kdtreequery( ...
				models{j},                         ...
				single(train_matrix{j}),          ...
				single(test_matrix{i}),           ...
				'NumNeighbors', 2);

			% Match is made if first closest match is closer
			% by some threshold than the second closest match.
			for k = 1:size(indeces, 2)
				if(distances(1, k) <= distances(2, k)*threshold)
					index_pairs = [index_pairs; k indeces(1, k)];
				end
			end

			if length(index_pairs) > maximum
				maximum = length(index_pairs);
				label = train_labels(j);
			end
		end
		labels(i) = label;
	end

	correct = find(labels == test_labels);
	accuracy = (length(correct) / length(test_labels)) * 100;
end
