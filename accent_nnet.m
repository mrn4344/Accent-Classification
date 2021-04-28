%% lkjsadlfkjlkasdf
clear all; close all;

load('knn_data.mat')
load('knn_data_gt.mat')

pop_limit = 10000;
numbers = zeros(length(unique(gt)),1);
to_delete = zeros(size(gt));

%balance
for i = 1:length(knn_data)
	if(numbers(gt(i)) < pop_limit)
		numbers(gt(i)) = numbers(gt(i)) + 1;
	else
		to_delete(i) = 1;
	end
	
	if(gt(i) ~= 5 && gt(i) ~= 15)
		to_delete(i) = 1;
	end
end

x = knn_data(to_delete == 0,:);
y = gt(to_delete == 0,:);
y(y == 5) = 1;
y(y == 15) = 2;
max = 0;
max_i = 0;

%% stuff
for i = 1:200
	options.method = 'nnet';
	options.nnet_hiddenLayerSize = i;
	[confusionMatrix_nnet1,accuracy_nnet1] =  classify677_hwk7(x,y,options);

	if accuracy_nnet1 > max
		max = accuracy_nnet1;
		max_i = i;
	end
end

fprintf("Max accuracy %g, at %d\n",max, max_i);