###############################################################################################
# MODULE: Module for various manual feature selection/modification
# FUNCTIONS - 
# 1. CORESET SAMPLING - Function to perform importance-sampling method called coresets
# USAGE : data_sampled = coreset(data)
# INPUT: data - N*d matrix where 'N' is the number of data points and 'd' is the number of features 
# OUTPUT: data_sampled - Sampled matrix with dimension M*d where 'M' is the number of sampled points 	
#						 and 'd' is the number of the features. 'M' is dependent on the feature set.
#
# 2. NAIVE SAMPLING - Function for randomly sampling data points from the dataset
# USAGE: data_sampled = naive(data,M)
# INPUT: data - N*d matrix where 'N' is the number of data points and 'd' is the number of features
#		 M - M is the number of data points to be sampled
# OUTPUT: data_sampled - Sampled matrix with dimension M*d where 'M' is the number of sampled points 	
#						 and 'd' is the number of the features.	
#

module sampling
export naive,coreset

import StatsBase

function coreset(data)
	features = data[:,1:end-1]
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

function naive(data, M)
	sampled = StatsBase.sample(1:size(data,1), M)
	data_sampled = data[sampled,:]
	return data_sampled
end

end