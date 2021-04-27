clear all;
%read data
data = readtable('cv-valid-train.csv');
accents = unique(data(:,7))

%remove non-accented data
data(strcmp(data.accent,''),:) = [];
accents = unique(data(:,7));

filter = [1 -0.9375];

testdata = readtable('cv-valid-test.csv');
testdata(strcmp(testdata.accent,''),:) = [];

knn_data = [];

%change accents to index
for i = 1:size(accents,1)
	replacedata = strcmp(data.accent,char(accents.accent(i)));
	data.accents(replacedata) = i;
	
	replacedata = strcmp(testdata.accent,char(accents.accent(i)));
	testdata.accents(replacedata) = i;
end
knn_data = zeros(10000,48000);
tic
for i = 1:10000
	y = load_train_fft(data.filename(i));
	y = y(1:48000);
	knn_data(i,:) = y';
	if mod(i,50) == 0
		toc
		fprintf('sample %d processed\n',i);
		tic
	end
end
toc

%setup matrix
c_mat = zeros(size(accents,1),size(accents,1));

%test?
for i = 1:100
	y = abs(fft(audioread("./cv-valid-test/" + char(testdata.filename(i)))));
	y = y(1:48000)';
	
	guessidx = knnsearch(knn_data, single(y));
	c_mat(data.accents(guessidx), testdata.accents(i)) = c_mat(data.accents(guessidx), testdata.accents(i)) + 1;
end

acc = sum(diag(c_mat))/sum(sum(c_mat));
fprintf("accuracy = %f\n",acc);

% filename1 = "./cv-valid-train/" + char(data.filename(1))
% filename2 = "./cv-valid-train/" + char(data.filename(2))
% 
% subplot(1,2,1)
% x = audioread("./cv-valid-train/" + char(data.filename(1)));
% x = conv(x, filter);
% plot(abs(fft(x)))
% 
% subplot(1,2,2)
% x = audioread("./cv-valid-train/" + char(data.filename(5)));
% x = conv(x,filter);
% plot(abs(fft(x)))

% subplot(2,2,1);
% plot(load_train_fft(data.filename(1)));
% 
% subplot(2,2,2);
% plot(load_train_fft(data.filename(2)));
% 
% subplot(2,2,3);
% plot(load_train_fft(data.filename(3)));
% 
% subplot(2,2,4);
% plot(load_train_fft(data.filename(4)));

% y = load_train_fft(data.filename(3));
%[y,fs] = audioread("./cv-valid-train/" + char(data.filename(1)));
%Y = fft(y);
%plot(0:abs(Y))


% [y,fs] = audioread("./cv-valid-train/" + char(data.filename(1)));
% windowSize = 256;
% windowOver = [];
% freqRange = 0:fs;
% spectrogram(y(:,1), windowSize, windowOver, freqRange, fs, 'xaxis');

% for i = 1:100
% 	%subplot(25,4,i)
% 	%plot(load_train_fft(data.filename(i)))
% 	[y,fs] = audioread("./cv-valid-train/" + char(data.filename(i)));
% 	fprintf('fs = %f\n', fs);
% end

