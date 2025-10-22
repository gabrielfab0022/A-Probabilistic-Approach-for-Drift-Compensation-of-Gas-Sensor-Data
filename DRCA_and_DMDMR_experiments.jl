
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


######DRCA#######

function DRCA(Xs, Xt, lambd, d)
    mu_s = mean(Xs, dims = 2)
    mu_t = mean(Xt, dims = 2)
    dim = size(Xs, 1)
    A = pinv((mu_s-mu_t)*(mu_s-mu_t)')*(Xs*Xs'+lambd*Xt*Xt')
    eigvals, P = eigen(A)
    eigvals = real.(eigvals)
    order = sortperm(eigvals)
    P = real.(P)
    P = P[:, order]
    P = P[:, end-d+1:end]
    return P 
end

#######DMDMR##########
function DMDMR(Xs, Xt, labels_xs, alpha, beta, gamma, d)
    mu_s = vec(mean(Xs, dims = 1))
    mu_t = vec(mean(Xt, dims = 1))
    m = length(unique(labels_xs))
    Ns = size(Xs,1)
    Nt = size(Xt,1)
    dim = size(Xs,2)
    Ts = zeros(m,Ns)
    Ts[collect(0:1:Ns-1)*m + labels_xs] .= 1
    H = Diagonal(ones(Ns)) - (1/Ns)*ones(Ns, Ns)
    Z = Ts*H*Xs
    M = Xs'*Xs + alpha*Xt'*Xt 
    M = M + beta*Z'*Z 
    M = M - gamma*(mu_s - mu_t)*(mu_s - mu_t)'
    eigvals, P = eigen(M)
    eigvals = real.(eigvals)
    order = sortperm(eigvals)
    P = real.(P)
    P = P[:, order]
    P = P[:, end-d+1:end]
    return P 
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


###REMOVENDO OUTLIERS DA BASE DE DADOS

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

accuracies_raw_data = []
accuracies_drca_data = []
accuracies_dmdmr_data = []

for val_set in 1:size(validation_idxs,1)
    println(val_set)
    
    #separating validation data
    
    val_set_raw_data = X[validation_idxs[val_set, 1]:validation_idxs[val_set,2],:]
    val_labels = labels[validation_idxs[val_set,1]:validation_idxs[val_set,2]]
    outputs_raw_data = knn_classifier(val_set_raw_data, training_set_raw, training_labels, 3)

    push!(accuracies_raw_data, mean(outputs_raw_data.==val_labels))


    #TESTING NOW WITH DRCA
    lambd_grid = [1e-3, 1e-2, 1e-1, 10e0, 1e+1, 1e+2, 1e+3, 1e+4]
    d_grid = [2, 4, 6, 8, 10, 12, 14, 16, 18]
    best_acc = 0
    for lambd in lambd_grid
        for d in d_grid
            #DATA NORMALIZATION TECHNIQUE FOUND IN THE AUTHORS' CODE
            source_domain = training_set_raw./sqrt.(sum(training_set_raw.^2, dims = 1))
            target_domain = val_set_raw_data./sqrt.(sum(val_set_raw_data.^2, dims = 1))
            P = DRCA(source_domain', target_domain', lambd, d)
            train_new = source_domain*P 
            val_new = target_domain*P 
            predictions = knn_classifier(val_new, train_new, training_labels, 3)
            acc = mean(predictions.==val_labels)
            if acc>best_acc
                best_acc = acc
            end
        end
    end
    push!(accuracies_drca_data, best_acc)

    #TESTING NOEW WITH DMDMR
    alpha_grid = [1e-4, 1e-3, 1e-2, 1e-1, 10e0, 1e+1, 1e+2, 1e+3, 1e+4]
    beta_grid  = [1e-4, 1e-3, 1e-2, 1e-1, 10e0, 1e+1, 1e+2, 1e+3, 1e+4]
    gamma_grid  = [1e-4, 1e-3, 1e-2, 1e-1, 10e0, 1e+1, 1e+2, 1e+3, 1e+4]
    d_grid = [2, 4, 6, 8, 10, 12, 14, 16, 18]
    best_acc = 0
    for alpha in alpha_grid
        for beta in beta_grid
            for gamma in gamma_grid
                for d in d_grid
                    source_domain = training_set_raw./sqrt.(sum(training_set_raw.^2, dims = 1))
                    target_domain = val_set_raw_data./sqrt.(sum(val_set_raw_data.^2, dims = 1))
                    P = DMDMR(source_domain, target_domain, training_labels, alpha, beta, gamma, d)
                    train_new = source_domain*P 
                    val_new = target_domain*P 
                    predictions = knn_classifier(val_new, train_new, training_labels, 3)
                    acc = mean(predictions.==val_labels)
                    if acc>best_acc
                        best_acc = acc
                    end
                end
            end 
        end 
    end
    push!(accuracies_dmdmr_data, best_acc)
  
end

println("Accuracies raw data: ", accuracies_raw_data)
println("Accuracies drca data: ", accuracies_drca_data)
println("Accuracies dmdmr data: ", accuracies_dmdmr_data)
