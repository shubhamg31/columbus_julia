module sampling
export naive,coreset

import StatsBase

function coreset(data, fs)
	features = data[:,fs]
	label = data[:,end]
	
	m = *(transpose(features),features)
	inv_features = inv(m)
	sensitivity1 = sum((*(features,inv_features) .* features), 1)
	sensitivity = Float64[x for x in sensitivity1]
	weight = StatsBase.WeightVec(sensitivity)
	sampled = StatsBase.sample(1:length(sensitivity), weight, int(2*(sum(sensitivity)-1)*100))
	if length(sampled) < size(data,1)
		data_sampled = data[sampled,:]
	else
		data_sampled = data	
	end
	println(size(data_sampled,1))
	return data_sampled
end

function naive(data, n)
	sampled = StatsBase.sample(1:size(data,1), n)
	data_sampled = data[sampled,:]
	return data_sampled
end

end