clear all;

%constants
train_path = "./cv-valid-train/";
test_path = "./cv-valid-test/";

win = hann(1024, 'periodic');


%read data
data = readtable('cv-valid-train.csv');
accents = unique(data(:,7));

%remove non-accented data
data(strcmp(data.accent,''),:) = [];
accents = unique(data(:,7))

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

knn_data = zeros(10000,14*4);
tic
for i = 1:10000
	[y,fs] = audioread(train_path + data.filename(i));
	bound = detectSpeech(y,fs);
	y = y(bound(1):bound(2));
	s = stft(y,'window',win,'overlaplength',512,'centered',false);
	coeffs = mfcc(s,fs);
	y = [mean(coeffs) var(coeffs) std(coeffs) kurtosis(coeffs)];
	%y = conv(y,filter);
	%y = sum(mfcc(y,fs));
	knn_data(i,:) = y;
	if mod(i,50) == 0
		toc
		fprintf('sample %d processed\n',i);
		tic
	end
end
toc

%% setup matrix
c_mat = zeros(size(accents,1),size(accents,1));

%test?
for i = 1:1000
	%y = abs(fft(audioread("./cv-valid-test/" + char(testdata.filename(i)))));
	%y = y(1:48000)';
	if i == 362
		continue
	end
	[y,fs] = audioread(test_path + testdata.filename(i));
	bound = detectSpeech(y,fs);
	y = y(bound(1):bound(2));
	
	s = stft(y,'window',win,'overlaplength',512,'centered',false);
	coeffs = mfcc(s,fs);
	% y = mean(abs(coeffs));
	y = [mean(coeffs) var(coeffs) std(coeffs) kurtosis(coeffs)];
	%y = conv(y,filter);
	%y = sum(mfcc(y,fs));
	
	guessidx = knnsearch(knn_data, y);
	c_mat(data.accents(guessidx), testdata.accents(i)) = c_mat(data.accents(guessidx), testdata.accents(i)) + 1;
end

acc = sum(diag(c_mat))/sum(sum(c_mat));
fprintf("accuracy = %f\n",acc);

