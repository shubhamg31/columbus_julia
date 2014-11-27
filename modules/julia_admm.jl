#####################################################################################################################
# MODULE : Implements Alternative Direction Method of Multipliers for Model fitting using Non linear loss functions
# 
# USAGE : model = admm(Data) 
# 
# INPUT : Data: M*(N+1) matrix 
#				It contains the data on which the model is to be fitted
#				For a given row, first N columns contain the features and last column 
#				contains actual labels/expected output
#
# OUTPUT : model: Model fitted on the data using the non-linear loss function(Log loss function)
#

module julia_admm
export admm

using julia_ls
using Roots

function root(fp, lo, hi)
	flo = fp(lo)
	fhi = fp(hi)
	if(flo > fhi)
		lo = lo + hi;
		hi = lo - hi;
		lo = lo - hi;

		flo = flo + fhi;
		fhi = flo - fhi;
		flo = flo - fhi;
	end

	while(flo*fhi > 0)
		if(flo > 0)
 	     	hi = lo;
  		    lo = lo*2;
    	else
	      	lo = hi;
      		hi = hi * 2;
  		end
	    flo = fp(lo);
    	fhi = fp(hi);
	end
	val = fzero(fp,lo,hi)
	return val
end

function admm(data)
	A = data[:,1:end-1]
	b = data[:,end]
	
	u = zeros(size(b))
	z = zeros(size(b))
	x = Cdouble[0 for i = 1:size(A,2)]

	LAMBDA = 0.001
	NEPOCH = 10

	Q,R = qr(A)

	for j = 1:NEPOCH
		println(j)
		#data = hcat(A,(z-u))
		x = R\(transpose(Q)*b)
		Ax = A*x
		for i = 1:size(z,1)
			lo = -10
			hi = 10
			fp(p) = -b[i] + 1.0/(1.0 + e^(-p)) + LAMBDA*(p-Ax[i]-u[i])
			z[i] = root(fp,lo,hi)
		end
		u = u + Ax - z
	end
	return(transpose(x))
end
end