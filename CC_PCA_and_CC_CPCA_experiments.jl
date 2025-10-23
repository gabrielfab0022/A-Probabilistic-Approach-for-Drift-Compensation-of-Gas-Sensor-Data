using Statistics
using LinearAlgebra
using Distributions

#####OUTLIER DETECTOR#####

function KL(X) #padrões nas linhas de X, retorna a base da transformada KL  
    Xc = (X.-mean(X, dims = 1))*(1/(size(X,1)-1))
    U, S, V = svd(Xc)
    pc_scores = S.^2
    pc_scores = pc_scores/sum(pc_scores)
    return V, pc_scores #autovetores da matriz de covariância nas colunas de V, autovalores em pc_score
end

function find_n(S, pct) #encontra quantas componentes são necessárias para reter pct da variância
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

function PCA(X, pct) #padrões nas linhas de X, pct da variancia
    V, S = KL(X)
    n = find_n(S, pct)
    basis = V[:, 1:n]
    X_red = X*basis
    return X_red #retorna X numa dimensão reduzida
end

function MAD(X) #median absolute deviation
    mad = 1.4826*median(abs.(X.-median(X, dims=1)), dims=1)
    return mad
end

function robust_sphere(X)
    X_star = (X.-median(X, dims=1))./(MAD(X).+eps())
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
    M = quantile(distribution, 0.25)
    c = quantile(distribution, 0.99)
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

#########JOINT DIAGONALIZATION########

function covariance_matrix(X) 
    N = size(X,1)
    X_centered = X.-mean(X, dims = 1)
    cov = (1/(N-1))*X_centered'*X_centered
    return cov
end

function off(list_matrices)
    acc = 0
    for i in 1:length(list_matrices)
        acc += sum((list_matrices[i]-Diagonal(list_matrices[i])).^2)
    end
    return acc
end

function norms_sum(list_matrices)
    acc = 0
    for i in 1:length(list_matrices)
        acc += norm(list_matrices[i])
    end
    return acc
end

function get_G(list_matrices, i, j)
    G = zeros(3,3)
    for k in 1:length(list_matrices)
        Ak = list_matrices[k]
        h = [(Ak[i,i]-Ak[j,j]) (Ak[i,j]+Ak[j,i]) 0]
        G += h'*h
    end
    return G
end


function get_R(list_matrices, i, j)
    N = size(list_matrices[1], 1)
    G = get_G(list_matrices, i, j)
    eigvals, eigvecs = eigen(G)
    idx = argmax(abs.(eigvals))
    x = real(eigvecs[1, idx])
    y = real(eigvecs[2, idx])
    z = real(eigvecs[3,idx])
    r = sqrt(x^2+y^2+z^2)
    s = y/(sqrt(2*r*(x+r)))
    c = sqrt((x+r)/(2*r))
    #println(c^2+s^2)
    #if (c^2+s^2)!= 1
    #    println("G: ", G)
    #    println("Eigvecs: ", eigvecs)
    #end
    R = Matrix{Float64}(I, N, N)
    R[i,i] = c
    R[j,j] = c
    R[i,j] = -s
    R[j,i] = s
    return R
end

function jacobi_algorithm(list_matrices, epsilon, max_iter) 
    n = size(list_matrices[1], 1) 
    idx = rand(1:length(list_matrices))
    Q = Matrix{Float64}(I, n, n)
    updated_matrices = copy(list_matrices)
    iter = 0;
    while (off(updated_matrices) > epsilon*norms_sum(updated_matrices)) && iter < max_iter 
        #println(off(updated_matrices))
         for i in 1:n
            for j in 1:n
                if  i != j 
                    #println(i)
                    #println(j)
                    R = get_R(updated_matrices, i, j)
                    Q = Q*R
                    for k in 1:length(updated_matrices)
                        #println(R)
                        #println(updated_matrices[k])
                        updated_matrices[k] = R'*updated_matrices[k]*R
                    end
                end
            end
        end
        iter += 1
        #println(iter)
        #println(off(updated_matrices))
    end
    return Q
end

function SIR(X, p) 
    p_matrix = ones(size(X,1))*p'
    p_scores = X*p;
    projections = p_matrix.*p_scores 
    e = X - projections 
    norms_X = sqrt.(sum(X.^2, dims = 2))
    norms_e = sqrt.(sum(e.^2, dims = 2))
    sir = mean(norms_X./norms_e)
    return sir
end

function find_sir_forQ(X, Q) 
    ratios = zeros(size(Q,2))
    for i in 1:size(Q,2)
        p = Q[:,i]
        ratios[i] = SIR(X, Q[:,i])
    end
    return ratios
end

######CLASSIFIER######

function dist(X, Y) #Dij = distance between Xi and Yj 
    D = sum(X.^2, dims = 2)*ones(1, size(Y, 1)) + ones(size(X,1))*sum(Y.^2, dims = 2)' - 2*X*Y'
    D = sqrt.(D.*(D.>=0))
end

function knn_classifier(test_data, training_data, training_labels, k) 
    distances = dist(test_data,training_data)
    outputs = []
    for i in 1:size(test_data,1)
        knn = sortperm(distances[i,:])[1:k]
        labels_nn = training_labels[knn]
        unique_labels_nn = unique(labels_nn)
        label_counts = [sum(labels_nn.==i) for i in unique_labels_nn]
        push!(outputs, unique_labels_nn[argmax(label_counts)])
    end
    return outputs
end 


#####LOADING DATASET##### 

####LOAD HERE YOUR DATASET


#####REMOVING OUTLIERS####
idx_outliers, w = PCOut(X);

to_keep = trues(size(X,1))
to_keep[idx_outliers] .= false

X = X[to_keep, :]
labels = labels[to_keep]

classes = ["ammonia 0.05%",  "propanoic acid 0.05%", "n-buthanol 0.1%", "ammonia 0.02%", "ammonia 0.01%", 
"n-buthanol 0.01%", "propanoic acid 0.02%", "propanoic acid 0.01%"]

labels = [findall(classes.==labels[i])[1] for i in 1:length(labels)]


n_samples = length(labels)


#EXPERIMENTS

training_size = 1000 #FOR T1

training_set_raw = X[1:training_size,:]
training_labels = labels[1:training_size]

validation_start_idx = collect(training_size+1:100:n_samples-training_size+1)
validation_idxs = hcat(validation_start_idx, validation_start_idx.+(training_size-1)) 
#1st column: starting index of each validation set; 2nd column: ending index of each validation set

#samples_i = samples from the i-th class
idx_samples_1 = findall(a-> a == 1, training_labels); samples_1 = training_set_raw[idx_samples_1,:]
idx_samples_2 = findall(a-> a == 2, training_labels); samples_2 = training_set_raw[idx_samples_2,:] 
idx_samples_3 = findall(a-> a == 3, training_labels); samples_3 = training_set_raw[idx_samples_3,:]
idx_samples_4 = findall(a-> a == 4, training_labels); samples_4 = training_set_raw[idx_samples_4,:]
idx_samples_5 = findall(a-> a == 5, training_labels); samples_5 = training_set_raw[idx_samples_5,:]
idx_samples_6 = findall(a-> a == 6, training_labels); samples_6 = training_set_raw[idx_samples_6,:]
idx_samples_7 = findall(a-> a == 7, training_labels); samples_7 = training_set_raw[idx_samples_7,:]
idx_samples_8 = findall(a-> a == 8, training_labels); samples_8 = training_set_raw[idx_samples_8,:]

#cov_i = covariance matrix of the samples from the i-th class
cov_1 = covariance_matrix(samples_1)
cov_2 = covariance_matrix(samples_2)
cov_3 = covariance_matrix(samples_3)
cov_4 = covariance_matrix(samples_4)
cov_5 = covariance_matrix(samples_5)
cov_6 = covariance_matrix(samples_6)
cov_7 = covariance_matrix(samples_7)
cov_8 = covariance_matrix(samples_8)

cov_matrices = [cov_1, cov_2, cov_3, cov_4, cov_5, cov_6, cov_7, cov_8]

Q = jacobi_algorithm(cov_matrices, 1e-6, 100)

snrs = find_sir_forQ(training_set_raw, Q) 
idx_max = argmax(snrs)

#drift component by joint diagonalization
p = Q[:,idx_max] #component with highest signal to noise ratio, as suggested in the paper

#drift component per class
p1 = KL(samples_1)[1]; p1 = p1[:,1]
p2 = KL(samples_2)[1]; p2 = p2[:,1]
p3 = KL(samples_3)[1]; p3 = p3[:,1]
p4 = KL(samples_4)[1]; p4 = p4[:,1]
p5 = KL(samples_5)[1]; p5 = p5[:,1]
p6 = KL(samples_6)[1]; p6 = p6[:,1]
p7 = KL(samples_7)[1]; p7 = p7[:,1]
p8 = KL(samples_8)[1]; p8 = p8[:,1]

P = [p1 p2 p3 p4 p5 p6 p7 p8]

accuracies_raw_data = []
accuracies_cccpca_data = []
accuracies_ccpca_data = []

for val_set in 1:size(validation_idxs,1)

    #separating validation data
    
    val_set_raw_data = X[validation_idxs[val_set, 1]:validation_idxs[val_set,2],:]
    val_labels = labels[validation_idxs[val_set,1]:validation_idxs[val_set,2]]
    outputs_raw_data = knn_classifier(val_set_raw_data, training_set_raw, training_labels, 3)

    push!(accuracies_raw_data, mean(outputs_raw_data.==val_labels))

    #CC-CPCA    
    train_new = (training_set_raw - ((training_set_raw)*p).*ones(size(training_set_raw,1))*p')
    val_new = (val_set_raw_data - ((val_set_raw_data)*p).*ones(size(val_set_raw_data,1))*p')

    predictions = knn_classifier(val_new, train_new, training_labels, 3)
    push!(accuracies_cccpca_data, mean(predictions.==val_labels))

    #CC-PCA
    best_accuracy = 0
    for i in 1:8 

        train_new = (training_set_raw - ((training_set_raw)*P[:,i]).*ones(size(training_set_raw,1))*P[:,i]')
        val_new = (val_set_raw_data - ((val_set_raw_data)*P[:,i]).*ones(size(val_set_raw_data,1))*P[:,i]')
        
        predictions = knn_classifier(val_new, train_new, training_labels, 3)
        acc = mean(predictions.==val_labels)
        if acc > best_accuracy
            best_accuracy = acc
        end 
    end
    push!(accuracies_ccpca_data, best_accuracy)
end

println("Accuracies raw data: ", accuracies_raw_data)
println("Accuracies cccpca data: ", accuracies_cccpca_data)
println("Accuracies ccpca data: ", accuracies_ccpca_data)