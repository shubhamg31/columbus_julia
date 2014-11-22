module stepadd_multicore
export add_feature

using julia_ls
using DataFrames

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

fs = [1,2,4,5]
full = [1:nfeat]
to_add = setdiff(full,fs)

function add_feature(A,fs,to_add)
	min_loss = Inf
	tasks = {A[:,union(fs,i)] for i in to_add}
	model = {Cdouble[0 for j = 1:(length(fs)+1)] for i = 1:length(to_add)}
	results = pmap(least_square, tasks, model)
	loss_value, idx = findmin([results[i][2] for i = 1:length(results)])
	model = results[idx][1]
	println(idx)
	return A[:,union(fs,idx)]
end

end