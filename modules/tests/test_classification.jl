push!(LOAD_PATH, "/home/shubham/Documents/research/columbus_julia/modules/")
using julia_perceptron
using julia_svm

f = open("results_classification.txt","w")

NCols=151
NRows=100000
d = rand(NRows, NCols)
fs = [1:NCols-1]

d = d[:,fs]

write(f, "Perceptron Algorithm Without Materialization: \n")
write(f, "Time Elapsed: ")
tic()
@time perceptron(d)
write(f,string(toc(),"\n"))

write(f,"\n")

write(f, "SVM Without Materialization: \n")
write(f, "Time Elapsed: ")
tic()
@time svm(d)
write(f,string(toc()))

close(f)
