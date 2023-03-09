import pickle
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
import os
import pickle
import cv2
import pandas as pd
import matplotlib.pyplot as plt

# Initializing control variable
debug = 1
list_save = 1
clip_flag = 1
clip_data = 1000
split_data = 0.8

# Define the learning rate
learning_rate = 0.1

# Define the number of iterations
num_iters = 1000
num_iters = 3
batch_size = 100
num_classes = 100

resize = 1
normalize = 1
width = 28
height = 28
dim = (width, height)
load_img = 1
show_img = 0

class Softmax:
        
    def forward(self, input_matrix):
        input_matrix = input_matrix - np.max(input_matrix, axis=1, keepdims=True)
        exp = np.exp(input_matrix)
        return exp / np.sum(exp, axis=1, keepdims=True)
    
    def loss(self, output, target):
        #train_lebel = np.eye(num_classes)[train_lebel.astype(int)]
        # error_output = output - train_lebel
        target_one_hot = np.zeros_like(output)
        target_one_hot[np.arange(output.shape[0]), target] = 1
        return -np.sum(target_one_hot * np.log(output)) / output.shape[0]

    
    def backward(self, output, target):
        # train_lebel = target
        # num_classes = output.shape[1]
        # train_lebel = np.eye(num_classes)[train_lebel.astype(int)]
        # return output - train_lebel
        
        target_one_hot = np.zeros_like(output)
        target_one_hot[np.arange(output.shape[0]), target] = 1
        # print("target_one_hot ", target_one_hot.shape)
        # print("output ", output.shape)
        return (output - target_one_hot) / output.shape[0]

class ReLU:
    def __init__(self):
        self.inputs = None

    def forward(self, x):
        self.inputs = x
        return x * (x > 0)

    def backward(self, grad_output):
        grad_input = grad_output * (self.inputs > 0)
        return grad_input

class DenseLayer:
    def __init__(self, output_dim):
        self.output_dim = output_dim
        self.weights = None
        self.bias = None
        self.input_matrix = None
        self.best_weights = None
        self.best_biases = None
        
    def forward(self, input_matrix):
        self.input_matrix = input_matrix
        if self.weights is None or self.bias is None:
            self.weights = np.random.randn(input_matrix.shape[1], self.output_dim) / np.sqrt(input_matrix.shape[1])
            self.bias = np.zeros((1, self.output_dim))
        return np.dot(input_matrix, self.weights) + self.bias
    
    def backward(self , grad_input, learning_rate):
        grad_weights = np.dot(self.input_matrix.T, grad_input)
        grad_bias = np.sum(grad_input, axis=0, keepdims=True)
        grad_input = np.dot(grad_input, self.weights.T)
        self.weights -= learning_rate * grad_weights
        self.bias -= learning_rate * grad_bias
        return grad_input , self.weights, self.bias

    def set_weights_biases(self, weights, biases):
        self.weights = weights
        self.bias = biases

    def save_weights_biases(self):
        self.best_weights = self.weights
        self.best_biases = self.bias

    def get_best_weights_biases(self):
        return self.best_weights, self.best_biases   

    def set_best_weights_biases(self):
        self.weights = self.best_weights
        self.bias = self.best_biases

class FlattenLayer:
    def __init__(self):
        self.input_shape = None
    
    def forward(self, input_matrix):
        self.input_shape = input_matrix.shape
        output_vector = input_matrix.reshape(-1, self.input_shape[1] * self.input_shape[2] * self.input_shape[3])
        return output_vector
    
    def backward(self, grad_input):
        return grad_input.reshape(self.input_shape)

class MaxPooling:
    def __init__(self, filter_dim, stride):
        self.filter_dim = filter_dim
        self.stride = stride
        self.input_shape = None
        self.relu_out = None
        self.X = None

    def forward(self, X):
        self.X = X
        N, H, W, C = X.shape
        self.input_shape = X.shape
        H_ = int(1 + (H - self.filter_dim) / self.stride)
        W_ = int(1 + (W - self.filter_dim) / self.stride)
        out = np.zeros((N, H_, W_, C))
        for n in range(N):
            for h in range(0, H - self.filter_dim + 1, self.stride):
                for w in range(0, W - self.filter_dim + 1, self.stride):
                    for c in range(C):
                        out[n, h // self.stride, w // self.stride, c] = np.max(
                            X[n, h:h + self.filter_dim, w:w + self.filter_dim, c])
        self.relu_out = out
        return out

    def backward(self, dout):
        N, H_out, W_out, C = dout.shape
        dX = np.zeros(self.input_shape)

        for n in range(N):
            for h_out in range(H_out):
                for w_out in range(W_out):
                    for c in range(C):
                        h_start = h_out * self.stride
                        h_end = h_start + self.filter_dim
                        w_start = w_out * self.stride
                        w_end = w_start + self.filter_dim

                        X_pool_region = self.X[n, h_start:h_end, w_start:w_end, c]
                        mask = X_pool_region == np.max(X_pool_region)
                        dX[n, h_start:h_end, w_start:w_end, c] += mask * dout[n, h_out, w_out, c]

        return dX


class Convolution:
    def __init__(self, num_filters, filter_dim, stride, padding):
        self.num_filters = num_filters
        self.filter_dim = filter_dim
        self.stride = stride
        self.padding = padding
        self.weights = np.random.randn(num_filters, filter_dim, filter_dim) / np.sqrt(filter_dim * filter_dim )
        self.biases = np.zeros((num_filters, 1))
        self.input_data = None
        self.best_weights = np.random.randn(num_filters, filter_dim, filter_dim) / np.sqrt(filter_dim * filter_dim )
        self.best_biases = np.zeros((num_filters, 1))
    
    def forward(self, input_data):
        self.input_data = input_data
        n, h, w = input_data.shape
        output_h = (h - self.filter_dim + 2 * self.padding) // self.stride + 1
        output_w = (w - self.filter_dim + 2 * self.padding) // self.stride + 1
        output_data = np.zeros((n, output_h, output_w, self.num_filters))
        input_data_padded = np.pad(input_data, ((0, 0), (self.padding, self.padding), (self.padding, self.padding)), 'constant')
        for i in range(n):
            for j in range(self.num_filters):
                for k in range(output_h):
                    for l in range(output_w):
                        h_start = k * self.stride
                        h_end = h_start + self.filter_dim
                        w_start = l * self.stride
                        w_end = w_start + self.filter_dim
                        input_slice = input_data_padded[i, h_start:h_end, w_start:w_end]
                        output_data[i, k, l, j] = np.sum(input_slice * self.weights[j]) + self.biases[j]
        return output_data



    def backward(self, del_v, lr):
        n, h, w = self.input_data.shape
        n, output_h, output_w, num_filters = del_v.shape
        del_v_prev = np.zeros((n, h, w))
        del_w = np.zeros((self.num_filters, self.filter_dim, self.filter_dim))
        del_b = np.zeros((self.num_filters, 1))
        input_data_padded = np.pad(self.input_data, ((0, 0), (self.padding, self.padding), (self.padding, self.padding)), 'constant')
        del_v_padded = np.pad(del_v_prev, ((0, 0), (self.padding, self.padding), (self.padding, self.padding)), 'constant')
        for i in range(n):
            for j in range(self.num_filters):
                for k in range(output_h):
                    for l in range(output_w):
                        h_start = k * self.stride
                        h_end = h_start + self.filter_dim
                        w_start = l * self.stride
                        w_end = w_start + self.filter_dim
                        input_slice = input_data_padded[i, h_start:h_end, w_start:w_end]
                        del_v_padded[i, h_start:h_end, w_start:w_end] += self.weights[j] * del_v[i, k, l, j]
                        del_w[j] += input_slice * del_v[i, k, l, j]
                        del_b[j] += del_v[i, k, l, j]
        del_v_prev = del_v_padded[:, self.padding:-self.padding, self.padding:-self.padding]
        self.weights -= lr * del_w
        self.biases -= lr * del_b
        return del_v_prev , self.weights , self.biases

    def set_weights_biases(self, weights, biases):
        self.weights = weights
        self.biases = biases

    def save_weights_biases(self):
        self.best_weights = self.weights
        self.best_biases = self.biases

    def get_best_weights_biases(self):
        return self.best_weights, self.best_biases
    
    def set_best_weights_biases(self):
        self.weights = self.best_weights
        self.biases = self.best_biases

def load_images_2(folder_path, limit_image):
    images = []
    max_shape = None
    i = 0
    for filename in os.listdir(folder_path):
        i += 1
        if i > limit_image:
            break

        img = cv2.imread(os.path.join(folder_path, filename), cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            continue

        img = cv2.resize(img, (28, 28))
        img = cv2.bitwise_not(img)
        thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        img = thresh / 255.0
        if show_img == 1:
            plt.imshow(img, cmap='gray', interpolation='bicubic')
            plt.show()     
        images.append(img)

    return np.array(images)



def load_images(folder_path,limit_image):
    images = []
    max_shape = None
    i=0
    for filename in os.listdir(folder_path):
        i+=1
        if i > limit_image:
            break

        #img = cv2.imread(os.path.join(folder_path, filename), cv2.IMREAD_GRAYSCALE)
        #load and grayscale and resize
        if resize == 0:
            img = cv2.imread(os.path.join(folder_path, filename), cv2.IMREAD_GRAYSCALE)
        else :
            img = cv2.imread(os.path.join(folder_path, filename), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

        if normalize == 0:
            if img is not None:
                img = np.interp(img, (0, 255), (0, 1))
                if max_shape is None:
                    max_shape = img.shape
                else:
                    max_shape = (max(max_shape[0], img.shape[0]), max(max_shape[1], img.shape[1]))
                images.append(img)
        else:
            images.append(img)
    if normalize == 0:
        padded_images = []
        for img in images:
            padded_img = np.zeros(max_shape, dtype=np.float32)
            padded_img[:img.shape[0], :img.shape[1]] = img
            padded_images.append(padded_img)
        return np.array(padded_images)
    elif normalize == 1:
        mean = np.mean(images)
        std = np.std(images)
        images = (images - mean) / std
        img = images[0]
        return images


def load_data_lebel(folder_list, csv_filenames,limit_image):

    # Load the data from the folders

    data_list = []
    shape_bool = False
    if load_img == 0:
        folder0_shape = load_images(folder_list[0],limit_image).shape
    else:
        folder0_shape = load_images_2(folder_list[0],limit_image).shape
    for folder in folder_list:
        if load_img == 0:
            data = load_images(folder,limit_image)
        else:
            data = load_images_2(folder,limit_image)
        if folder != folder_list[0]:
            shape_diff = np.array(folder0_shape) - np.array(data.shape)
            if(shape_diff[0] == 0 and shape_diff[1] == 0 and shape_diff[2] == 0):
                shape_bool = False
            else:
                shape_bool = True
            padding = [(0,0)] + [(shape_diff[i] // 2, shape_diff[i] - shape_diff[i] // 2) for i in range(1, len(shape_diff))]
            data = np.pad(data, padding, mode='constant', constant_values=0)
        data_list.append(data)
        print("Data ", folder, " Shape: ", data.shape, " Re-Shape : ", shape_bool)

    data = np.concatenate(data_list, axis=0)
    print("Data Shape after concatenation: ", data.shape)

    print("Data merged successfully ... ")
    print("Data Index Zero : ", data[0])
    print("*"*50+"\n")

    labels_list = []

    # Loop over the csv_filenames to extract lebels
    for filename in csv_filenames:
        df = pd.read_csv(filename)
        labels = df["digit"].values
        labels_list.append(labels)
        print("Labels ", filename, " Shape: ", labels.shape)

    # Concatenate the labels_list into a single numpy array
    labels = np.concatenate(labels_list)
    print("Labels Shape: ", labels.shape)
    print("Traing lebels loaded successfully ... ")
    print("Data Lebel Zero : ", labels[0])
    print("*"*50+"\n")

    return data, labels[:limit_image]



if __name__ == '__main__':
    #My name and id
    #Fahmid and 1705087
    print("\n"+"*"*50)
    print("Name: Bangla Character Recognition Challenge ")
    print("Student ID: 1705087 ")
    print("*"*50+"\n")


    # train_folder_list = ['D:\\Maf Chai Vai Jayga Nai\\Data_CNN\\training-a',
    #                'D:\\Maf Chai Vai Jayga Nai\\Data_CNN\\training-b',
    #                'D:\\Maf Chai Vai Jayga Nai\\Data_CNN\\training-c']

    # train_csv_filenames = ['D:\\Maf Chai Vai Jayga Nai\\Data_CNN\\training-a.csv',
    #             'D:\\Maf Chai Vai Jayga Nai\\Data_CNN\\training-b.csv',
    #             'D:\\Maf Chai Vai Jayga Nai\\Data_CNN\\training-c.csv']

    train_folder_list = ['D:\\Maf Chai Vai Jayga Nai\\Data_CNN\\training-a']
    train_csv_filenames = ['D:\\Maf Chai Vai Jayga Nai\\Data_CNN\\training-a.csv']
    
    print("Loading Training Data ... ")
    print("*"*50+"\n")

    # Load the train data and train lebel
    train_data , train_lebel = load_data_lebel(train_folder_list, train_csv_filenames,limit_image=clip_data)

    test_folder_list = ['D:\\Maf Chai Vai Jayga Nai\\Data_CNN\\training-d']
    test_csv_filenames = ['D:\\Maf Chai Vai Jayga Nai\\Data_CNN\\training-d.csv']

    print("Loading Testing Data ... ")
    print("*"*50+"\n")

    # Load the test data and test lebel
    test_data , test_lebel = load_data_lebel(test_folder_list, test_csv_filenames,limit_image=clip_data)

    print("Intializing model ...")
    print("*"*50+"\n")

    conv_layer = Convolution(num_filters=16, filter_dim=3, stride=2, padding=0)
    relu_layer = ReLU()
    pool_layer = MaxPooling(filter_dim=2, stride=2)
    flatten_layer = FlattenLayer()
    dense_layer = DenseLayer(output_dim= num_classes)
    softmax_layer = Softmax()

    print("conv_layer one =  num_filters : ", conv_layer.num_filters , " filter_dim : ", conv_layer.filter_dim , " stride : ", conv_layer.stride , " padding : ", conv_layer.padding)
    print("max pooling one = filter_dim : ", pool_layer.filter_dim , " stride : ", pool_layer.stride)
    print("dense_layer one = output_dim : ", dense_layer.output_dim)
    print("softmax_layer one = num_classes : ", num_classes)

    print("\ntrain_data : ", train_data.shape)
    print("train_lebel : ", train_lebel.shape)
    print("test_data : ", test_data.shape)
    print("test_lebel : ", test_lebel.shape)

    # train and test with 1000 images
    if clip_flag:
        train_data = train_data[:clip_data, :, :]
        train_lebel = train_lebel[:clip_data]
        test_data = test_data[:clip_data, :, :]
        test_lebel = test_lebel[:clip_data]

    print("\nclipped train_data : ", train_data.shape)
    print("clipped train_lebel : ", train_lebel.shape)
    print("clipped test_data : ", test_data.shape)
    print("clipped test_lebel : ", test_lebel.shape)

    # print("train_lebel : ", train_lebel)
    # print("test_lebel : ", test_lebel)

    # split the data into train and validation
    if split_data:
        train_data_all = train_data
        train_lebel_all = train_lebel
        split = int(split_data * train_data_all.shape[0])
        train_data = train_data_all[:split, :, :]
        train_lebel = train_lebel_all[:split]
        val_data = train_data_all[split:, :, :]
        val_lebel = train_lebel_all[split:]


        print("\nsplit train_data : ", train_data.shape)
        print("split train_lebel : ", train_lebel.shape)
        print("split val_data : ", val_data.shape)
        print("split val_lebel : ", val_lebel.shape)

    


    print("\nnum_iters : ", num_iters)
    print("batch_size : ", batch_size)
    print("learning_rate : ", learning_rate)
    print("clip_data no : ", clip_data)
    print("list_save : ", list_save)

    # Define the loss list
    loss_list = []
    val_loss_list = []

    all_accuracy = []
    val_accuracy = []

    all_f1_score = []
    val_f1_score = []

    print("\nRunning CNN ... ")
    print("*"*50+"")

    max_accuracy = 0
    max_f1_score = 0
    iter_loss = 0
    iter_accuracy = 0
    iter_f1_score = 0

    # Train the network
    if list_save:
        for i in range(num_iters):
            iter_loss = 0
            # Split the training data into batches
            for batch in range(0, len(train_data), batch_size):
                batch_data = train_data[batch:batch + batch_size]
                batch_labels = train_lebel[batch:batch + batch_size]

                # Forward Propagation
                conv_out = conv_layer.forward(batch_data)
                relu_out = relu_layer.forward(conv_out)
                pool_out = pool_layer.forward(relu_out)
                flatten_out = flatten_layer.forward(pool_out)
                dense_out = dense_layer.forward(flatten_out)
                output = softmax_layer.forward(dense_out)

                # Compute the loss
                loss = softmax_layer.loss(output, batch_labels)
                iter_loss += loss

                train_softmax_out = np.argmax(output, axis=1)
                accuracy = np.mean(train_softmax_out == batch_labels)
                f1 = f1_score(batch_labels, train_softmax_out, average='macro')

                iter_accuracy += accuracy
                iter_f1_score += f1

                # Compute the derivatives
                d_out = softmax_layer.backward(output, batch_labels)
                grad_input, dense_weight, dense_bias = dense_layer.backward(d_out, learning_rate)
                d_flatten = flatten_layer.backward(grad_input)
                d_pool = pool_layer.backward(d_flatten)
                d_relu = relu_layer.backward(d_pool)
                d_conv, convo_weight, convo_bias = conv_layer.backward(d_relu, learning_rate)

                
                print("Iteration: ", i, "Batch: ", batch // batch_size, "Loss: ", loss)
            
            iter_loss = iter_loss / (len(train_data) / batch_size)
            loss_list.append(iter_loss)

            val_conv_out = conv_layer.forward(val_data)
            val_relu_out = relu_layer.forward(val_conv_out)
            val_pool_out = pool_layer.forward(val_relu_out)
            val_flatten_out = flatten_layer.forward(val_pool_out)
            val_dense_out = dense_layer.forward(val_flatten_out)
            output = softmax_layer.forward(val_dense_out)

            loss = softmax_layer.loss(output, val_lebel)
            val_loss_list.append(loss)

            val_softmax_out = np.argmax(output, axis=1)
            accuracy = np.mean(val_softmax_out == val_lebel)
            f1 = f1_score(val_lebel, val_softmax_out, average='macro')
            print("\nIteration: ", i, " Accuracy: ", accuracy*100, "%")
            print("Iteration: ", i, " Final F1 Score: ", f1)
            print("Training Loss: ", iter_loss)
            print("Validation Loss: ", loss)
            print("\n")

            val_accuracy.append(accuracy)
            val_f1_score.append(f1)

            all_accuracy.append(iter_accuracy / (len(train_data) / batch_size))
            all_f1_score.append(iter_f1_score / (len(train_data) / batch_size))

            

            if accuracy > max_accuracy:
                max_accuracy = accuracy

            if f1 > max_f1_score:
                max_f1_score = f1
                conv_layer.save_weights_biases()
                dense_layer.save_weights_biases()

    else:
        with open('1705087_model.pickle', 'rb') as f:
            [convo_weight, conv_bias, dense_weight, dense_bias] = pickle.load(f)

        # print("convo_weight : ", convo_weight)
        # print("conv_bias : ", conv_bias)
        # print("dense_weight : ", dense_weight)
        # print("dense_bias : ", dense_bias)

        conv_layer.set_weights_biases(convo_weight, conv_bias)
        dense_layer.set_weights_biases(dense_weight, dense_bias)

    if list_save:
        print("\nMax Accuracy : ", max_accuracy*100, "%")
        print("Max F1 Score : ", max_f1_score)
        print("\nSaving the model weights and biases...")
        conv_layer.set_best_weights_biases()
        dense_layer.set_best_weights_biases()
    
            
    # Save the model
    if list_save:
        convo_weight , conv_bias = conv_layer.get_best_weights_biases()
        dense_weight , dense_bias = dense_layer.get_best_weights_biases()

        with open('1705087_model.pickle', 'wb') as f:
            pickle.dump([convo_weight, conv_bias, dense_weight, dense_bias], f)
    
        print("\nModel saved Successfully in model Pickle ...")

    test_conv_out = conv_layer.forward(test_data)
    test_relu_out = relu_layer.forward(test_conv_out)
    test_pool_out = pool_layer.forward(test_relu_out)
    test_flatten_out = flatten_layer.forward(test_pool_out)
    test_dense_out = dense_layer.forward(test_flatten_out)
    test_softmax_out = softmax_layer.forward(test_dense_out)

    print("\nDimenstion of model ...")
    print("*"*50+"\n")

    print("conv_layer : ", test_conv_out.shape)
    print("relu_layer : ", test_relu_out.shape)
    print("pool_layer : ", test_pool_out.shape)
    print("flatten_layer : ", test_flatten_out.shape)
    print("dense_layer : ", test_dense_out.shape)
    print("softmax_layer : ", test_softmax_out.shape)

    print("\ntest_data : ", test_data.shape)
    print("test_lebel : ", test_lebel.shape)

    print("\nPerformance of model ...")
    print("*"*50+"\n")
        
    test_softmax_out = np.argmax(test_softmax_out, axis=1)
    print("\ntest_softmax_out.shape : ", test_softmax_out.shape)
    print("test_softmax_out : ", test_softmax_out)
    print("real test_lebel.shape : ", test_lebel.shape)
    print("test_lebel : ", test_lebel)
    
    #test_lebel = test_lebel(test_lebel, axis=1)
    accuracy = np.mean(test_softmax_out == test_lebel)
    print("\nFinal Accuracy: ", accuracy*100, "%")

    f1 = f1_score(test_lebel, test_softmax_out, average='macro')
    print("\nFinal F1 Score: ", f1)

    confusion_matrix = confusion_matrix(test_lebel, test_softmax_out)
    print("\nConfusion Matrix: \n", confusion_matrix)

    # Plot for Confusion Matrix
    plt.matshow(confusion_matrix)
    plt.title('Confusion Matrix Plot')
    plt.colorbar()
    plt.xlabel('True Label')
    plt.ylabel('Predicted Label')
    plt.show()

    # Plot for Loss
    plt.plot(range(len(loss_list)), loss_list, label='train loss')
    plt.plot(range(len(val_loss_list)), val_loss_list, label='validation loss')
    plt.xlabel('Epoc No.')
    plt.ylabel('Loss')
    plt.title('Learning Rate: '+str(learning_rate))
    plt.legend()
    plt.show()

    # Plot for Accuracy
    plt.plot(range(len(all_accuracy)), all_accuracy, label='train accuracy')
    plt.plot(range(len(val_accuracy)), val_accuracy, label='validation accuracy')
    plt.xlabel('Epoc No.')
    plt.ylabel('Accuracy')
    plt.title('Learning Rate: '+str(learning_rate))
    plt.legend()
    plt.show()

    # Plot for F1 Score
    plt.plot(range(len(all_f1_score)), all_f1_score, label='train f1 score')
    plt.plot(range(len(val_f1_score)), val_f1_score, label='validation f1 score')
    plt.xlabel('Epoc No.')
    plt.ylabel('F1 Score')
    plt.title('Learning Rate: '+str(learning_rate))
    plt.legend()
    plt.show()



    print("*"*50+"\n")


    

    


