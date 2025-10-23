import numpy as np
import sklearn 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import tensorflow as tf
from adapt.feature_based import DANN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers.legacy import Adam

#LOAD HERE YOUR DATASET 
#OUTLIERS SHOULD ALREADY BE REMOVED


train_size = 1000
keep = [5,7,8]

best_accuracies = []

dimensions = [8,16,32]
lambdas = [0.2, 0.5, 0.8, 1]

for i in range(16):
    val_number = i+1
    print('Validation: ', val_number)
    best_accuracy = 0 
    train_data = X[:train_size]
    train_labels = labels[:train_size].astype(int)
    val_data = X[train_size+(val_number-1)*100:train_size+(val_number-1)*100+train_size]
    val_labels = labels[train_size+(val_number-1)*100:train_size+(val_number-1)*100+train_size].astype(int)
    train_output_onehot = np.eye(8)[train_labels-1]
    val_output_onehot = np.eye(8)[val_labels-1]

    '''
    
    UNCOMMENT THESE LINES TO CONSIDER ONLY THE THREE BEST PERFORMING CLASSES 

    train_data = train_data[np.isin(train_labels, keep)]  
    train_labels = train_labels[np.isin(train_labels, keep)] 
    val_data = val_data[np.isin(val_labels, keep)] 
    val_labels = val_labels[np.isin(val_labels, keep)] 
    
    train_output_onehot = np.eye(3)[train_labels-1]
    val_output_onehot = np.eye(3)[val_labels-1]
    
    '''

    mean_train = train_data.mean(axis=0)
    std_train  = train_data.std(axis=0)
    std_train[std_train == 0] = 1
    mean_test = val_data.mean(axis=0)
    std_test  = val_data.std(axis=0)
    std_test[std_test == 0] = 1
    train_data = (train_data - mean_train) / std_train
    val_data  = (val_data - mean_test) / std_test

    #cont = 1
    
    for d in dimensions:
        for lamb in lambdas:
            #print(cont)
            #cont += 1
            tf.keras.backend.clear_session()
            encoder_network= Sequential([
                Input(shape=(18,)),
                Dense(100, activation='tanh'),
                Dense(d, activation='tanh')
            ])

            classifier_network = Sequential([
                Input(shape=(d,)),
                Dense(8, activation='softmax')
            ])

            discriminator_network = Sequential([
                Input(shape=(d,)),
                Dense(10, activation = 'tanh'),
                Dense(1, 'sigmoid')
            ])

            dann = DANN(
            encoder=encoder_network,
            task=classifier_network,
            discriminator = discriminator_network,
            Xt=val_data,
            optimizer=Adam(0.0001),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"],
            lambda_ = lamb,
            batch_size=16,
            epochs=1000,
            verbose=0
        )
                    
            dann.fit(train_data, train_output_onehot, verbose=0)
            
            x_train = dann.encoder.predict(train_data, verbose=0)
            x_val = dann.encoder.predict(val_data, verbose=0)
            knn = KNeighborsClassifier(n_neighbors=3)
            knn.fit(x_train, train_labels)
            y_pred = knn.predict(x_val)
            acc = accuracy_score(val_labels, y_pred)
            
            if acc > best_accuracy:
                best_accuracy = acc
    print("Best accuracy: ", best_accuracy)
    best_accuracies.append(best_accuracy)
    np.savetxt(path+'accuracies.txt', np.array(best_accuracies))