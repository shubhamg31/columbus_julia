### Adding Folder containing Modules to the path ###
push!(LOAD_PATH, "/home/shubham/Documents/research/columbus_julia/modules/")

### Importing Required Modules###
using julia_ls

############################# Synthetic Data #################################
### Initializing Data: Expected Output = Sum of features ###

NCols=3
NRows=1000
d = rand(NRows, NCols)
for row = 1:NRows
  d[row, NCols] = sum(d[row,1:end-1])
end

#Last Column is the Expected Output
#Selecting a set of features for model estimation
fs = [1:NCols-1]
d = d[:,union(fs,NCols)]
println("Data Initialised")

### Model Fitting using Least Square Regression###
model_ls, loss_ls = least_square(d)

predicted = zeros(size(d,1))
for i = 1:NRows
  predicted[i] = predicted[i] + sum(transpose(model_ls).*d[i,1:NCols-1])
end
### Output ###
println(hcat(d,predicted)[1:20,:])

############################# Real Data ####################################
### Reading Data from a CSV file ####
input_file = "/home/shubham/Documents/research/data/gasoline.csv"
data = readcsv(input_file)

##### Breaking the data into features and expected output and normalizing the data #####
feature = data[:, 2:4]
feature = (feature .- minimum(feature,1))./(maximum(feature,1) - minimum(feature,1))
output = data[:, 6]
output = (output .- minimum(output,1))./(maximum(output,1) - minimum(output,1))
data = hcat(feature,ones(size(feature,1)),output)

### Model Fitting using Least Square Regression###
model_ls, loss_ls = least_square(data)

### Output ###
println(model_ls)