function x = load_train_fft(filename)
	x = abs(fft(audioread("./cv-valid-train/" + filename,'native'),48000));
end