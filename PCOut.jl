#############################################################################################
#This code contains an implementation of the outlier detection algorithm proposed in the paper 
#"Outlier identification in high dimensions", written by P Filzmoser, R Maronna and M Werner, 
#and published in 2008
#############################################################################################


using Statistics
using Distributions
using LinearAlgebra
using PyPlot
pygui(true)
close("all")


function KL(X) #this function takes as input the data matrix (each row is a data point) and outputs the basis of the KL Transform
    X = X.-mean(X, dims = 1)
    _, S, V = svd(X)
    pc_scores = S.^2
    return V, pc_scores #the columns of V are the eigenvectors of the covariance matrix
end 

function find_n(S, pct) #find the number of components needed to retain pct% of the total variance 
    S_sorted =  reverse(sort(S))
    total_var = sum(S_sorted)
    n = 0
    acc_var = 0
    while acc_var < pct
        n = n+1
        acc_var = acc_var+S_sorted[n]/total_var
    end
    return n
end

function PCA(X, pct) #apply PCA keeping the number of components required to retain pct% of the variance
    V, S = KL(X)
    n = find_n(S, pct)
    basis = V[:, 1:n]
    X_red = X*basis
    return X_red #returns the projection of X onto a reduced dimension
end

function MAD(X) #median absolute deviation
    mad = 1.4826*median(abs.(X.-median(X, dims=1)), dims=1)
    return mad
end

function robust_sphere(X)
    X_star = (X.-median(X, dims=1))./MAD(X)
    return X_star
end

function kurtosis_weights(X)
    aux = (X.-median(X, dims=1)).^4
    aux = aux./(MAD(X).^4)
    aux = sum(aux, dims=1)
    aux = aux/(size(X,1))
    w = abs.(aux.-3)
    w = w/sum(w)
    return vec(w)
end

function weighted_norm(X, w)
    weighted_X = X.*w';
    RD = sqrt.(sum(weighted_X.^2, dims=2))
    return RD
end

function distances_transform(RD, p)
    distribution = Chisq(p)
    q = quantile(distribution, 0.5)
    d = RD*sqrt(q)/(median(RD))
    return d
end

function translated_biweight_w2(d, p)
    distribution = Chisq(p)
    M = sqrt(quantile(distribution, 0.25))
    c = sqrt(quantile(distribution, 0.99))
    w2i = 0*d
    for i in 1:length(d)
        if d[i]<= M
            w2i[i] = 1
        elseif d[i] < c
            w2i[i] = (1-((d[i]-M)/(c-M))^2)^2
        else 
            w2i[i] = 0
        end
    end
    return w2i
end

function translated_biweight_w1(d)
    M = quantile(d, 1/3)
    c = median(d)+2.5*MAD(d)[1]
    w1i = 0*d
    for i in 1:length(d)
        if d[i]<= M
            w1i[i] = 1
        elseif d[i] < c
            w1i[i] = (1-((d[i]-M)/(c-M))^2)^2
        else 
            w1i[i] = 0
        end
    end
    return w1i
end

function PCOut(X)
    #step 1
    X_star = robust_sphere(X)
    #step 2
    Z = PCA(X_star, 0.99)
    Z_star = robust_sphere(Z)
    #step 3 
    w = kurtosis_weights(Z_star)
    RD = vec(weighted_norm(Z_star, w))
    d = distances_transform(RD, size(Z_star,2))
    #step 4
    w1i = translated_biweight_w1(d)
    #step 5
    d = weighted_norm(Z_star, ones(size(Z_star,2)))
    w2i = translated_biweight_w2(d, size(Z_star, 2))
    #step 6 
    s = 0.25
    w = vec((w1i.+s).*(w2i.+s)/(1+s)^2)
    index_outliers = findall(a-> a<0.25, w)
    return index_outliers, w
end

#CREATING THE SAME DATASET THAT THE AUTHORS USE TO TEST THEIR ALGORITHM
p = 20 #dimension
non_outliers = randn(1000, p);

k = 5
delta = 1
cov_matrix = delta*Diagonal(ones(p))
a0 = rand(p)
a0 = a0.-mean(a0)
a0 = a0/norm(a0)

outliers = randn(100,p)*cov_matrix'.+k*a0'

X = vcat(non_outliers, outliers)

pho = 0.5
R = Diagonal(ones(p))+(pho*ones(p,p)-Diagonal(ones(p)*pho))
X = X*R

idx_outliers, _ = PCOut(X);

FN = (100-sum(idx_outliers.>1000))/100
FP = sum(idx_outliers.<=1000)/1000

println("False negative rate: ", FN*100, "%")
println("False positive rate: ", FP*100, "%")








