##### Refer http://julialang.org/blog/2013/04/distributed-numerical-optimization/ ######

module stepdrop_multicore
export drop_feature_multicore

using julia_ls

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

A = examples
fs = A[1,1:end-1]

function drop_feature_multicore(A, fs)
	min_loss = Inf
	
	tasks = {A[:,[1:i-1,i+1:size(A,2)]] for i = 1:length(fs)}
	model = {Cdouble[0 for j = 1:(length(fs)-1)] for i = 1:length(fs)}
	
	results = pmap(least_square, tasks, model)
	loss_value, idx = findmin([results[i][2] for i = 1:length(results)])
	model = results[idx][1]

	A_drop = A[:,[1:idx-1,idx+1:size(A,2)]]
	println(idx)
	return A_drop
end
end
#A_drop = stepdrop_multicore(A,fs)
