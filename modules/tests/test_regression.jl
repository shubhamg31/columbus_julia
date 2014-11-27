push!(LOAD_PATH, "/home/shubham/Documents/research/columbus_julia/modules/")
using julia_ls
using julia_lr
using julia_admm

f = open("results.txt","w")

NCols=151
NRows=100000
d = rand(NRows, NCols)
fs = [1:NCols-1]

d = d[:,fs]

write(f, "Least Squares Regression Without Materialization: \n")
write(f, "Time Elapsed: ")
tic()
@time least_square(d)
write(f,string(toc(),"\n"))

write(f,"\n")

write(f, "Logistic Regression Without Materialization: \n")
write(f, "Time Elapsed: ")
tic()
@time logit_reg(d)
write(f,string(toc()))

write(f,"\n")

write(f, "Logistic Regression Using ADMM: \n")
write(f, "Time Elapsed: ")
tic()
@time admm(d)
write(f,string(toc()))
close(f)
