# ---------------------------------------------------------------------------
# Source Code for:
# "A Probabilistic Approach for Drift Compensation of Gas Sensor Data"
# Author(s): Gabriel F.A. Bastos, Jugurta Montalvão and Luiz Miranda
# ---------------------------------------------------------------------------
# This code implements the proposed drift compensation method described in the paper.
#
# IMPORTANT NOTE:
# The dataset used in the experiments was provided by a collaborator and
# cannot be publicly redistributed due to data-sharing restrictions.
# Therefore, the dataset is not included in this repository.
#
# Researchers who have access to the same dataset can reproduce the results
# using the provided code. Alternatively, the implementation can be adapted
# to other datasets with similar structure.
# ---------------------------------------------------------------------------


using DataFrames
using CSV
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


######PROJECTION ONTO THE ZETA DOMAIN######

function project_to_zeta(X, K) #X: Data matrix; K: window size
    n_points = size(X,1)
    n_sensors = size(X,2)
    X_zeta = zeros(n_points, n_sensors)
    
    for j in 1:n_sensors
        window = X[1:K, j]
        for i in 1:K
            X_zeta[i,j] = sum(window .<= X[i,j])/K 
        end 
    end 
    
    for j in 1:n_sensors
        for i in K+1:n_points 
            window = X[i-K+1:i, j]
            X_zeta[i,j] = sum(window .<= X[i,j])/K
        end
    end 

    return X_zeta
end


######CLASSIFIER######

function dist(X, Y) #Dij = distance between Xi and Yj 
    D = sum(X.^2, dims = 2)*ones(1, size(Y, 1)) + ones(size(X,1))*sum(Y.^2, dims = 2)' - 2*X*Y'
    D = sqrt.(D.*(D.>=0))
end

function knn_classifier(test_data, training_data, training_labels, k) #
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

function accuracy(ground_truth, predictions)
    acc = sum(ground_truth.==predictions)/length(predictions)
    return acc
end



#####LOADING DATASET##### 

####LOAD HERE YOUR DATASET

#X: DATA MATRIX (n_points x 17 for the case of the dataset used in the paper); this dataset must be time-ordered!!!
#labels: vector with the labels

data = CSV.read("C:\\Users\\Gabriel Francisco\\Desktop\\TCC\\BrazilData\\data_180s.csv", DataFrame)

X = Matrix(data[:,2:33])
labels = data[:,end]
days = data[:,1]

labels_unique = unique(labels)

to_remove = [7, 8, 9, 10, 11, 16, 17, 18, 22, 23, 24, 25, 26,29]

to_keep = trues(size(X,2))
to_keep[to_remove] .= false

X = X[:, to_keep]

#####REMOVING OUTLIERS####
idx_outliers, w = PCOut(X);

to_keep = trues(size(X,1))
to_keep[idx_outliers] .= false

X = X[to_keep, :]
labels = labels[to_keep]


n_samples = length(labels)

#####PROJECTING DATA ONTO ZETA-DOMAIN#####

K = 100
X_zeta = project_to_zeta(X, K)

#####TRAINING AND TESTING CLASSIFIER####

training_size = 1000 

training_set_raw = X[1:training_size,:]
training_set_zeta = X_zeta[1:training_size,:]
training_labels = labels[1:training_size]

classes = ["ammonia 0.05%",  "propanoic acid 0.05%", "n-buthanol 0.1%", "ammonia 0.02%", "ammonia 0.01%", 
"n-buthanol 0.01%", "propanoic acid 0.02%", "propanoic acid 0.01%"]

validation_start_idx = collect(training_size+1:100:n_samples-training_size+1)
validation_idxs = hcat(validation_start_idx, validation_start_idx.+(training_size-1)) 
#1st column: starting index of each validation set; 2nd column: ending index of each validation set

n_val_sets = size(validation_idxs, 1) #number of validation sets 
accuracies_raw_data = []
accuracies_zeta_data = []

for val_set in 1:n_val_sets
    #separating validation data
    val_set_raw_data = X[validation_idxs[val_set, 1]:validation_idxs[val_set,2],:]
    val_set_zeta_data = X_zeta[validation_idxs[val_set, 1]:validation_idxs[val_set,2],:]
    val_labels = labels[validation_idxs[val_set,1]:validation_idxs[val_set,2]]

    #testing classification 
    outputs_raw_data = knn_classifier(val_set_raw_data, training_set_raw, training_labels, 3)
    outputs_zeta_data = knn_classifier(val_set_zeta_data, training_set_zeta, training_labels, 3)

    push!(accuracies_raw_data, mean(outputs_raw_data.==val_labels))
    push!(accuracies_zeta_data,mean(outputs_zeta_data.==val_labels))
end 

println("Accuracies raw data: ", accuracies_raw_data)
println("Accuracies zeta data: ", accuracies_zeta_data)


