module stepdrop

export drop_feature

using julia_ls
using julia_lr
using julia_perceptron
using julia_svm

nexp = 100
nfeat = 10

examples = Array(Cdouble, nexp, nfeat+1)
for row = 1:nexp
	for col = 1:nfeat
		examples[row, col] = rand()
	end
	if rand() > 0.8
		examples[row, nfeat+1] = 0
	else
		examples[row, nfeat+1] = 1
	end
end

###model = Cdouble[0 for i = 1:length]
###model = ls(examples, model)

fs = [1:nfeat]
A = examples
b = examples[:,end]

function drop_feature(func,A,fs)
	min_loss = Inf
	loss_value = Array(Float32, length(fs))
	idx = 0
	for i = 1:length(fs)
		data = A[:,[1:i-1,i+1:end]]
		model = Cdouble[0 for i = 1:size(data,2)]
		model, loss_value = func(data, model)
		println(loss_value)
		if loss_value < min_loss
			min_loss = loss_value
			final = model
			idx = i
		end
	end
	println(idx)
	return A[:,[1:idx-1,idx+1:end]]
end

end
