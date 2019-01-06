import numpy as np
import pickle
import os
import sys
from scipy import sparse
import tensorflow as tf


def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

def convert_onehot(values):
    a = np.zeros((values.size,values.max() + 1))
    a[np.arange(values.size),values] = 1
    return a.T

class DeepFeedForwardNetwork(object):

    def __init__(self,x,y,sizes,weight='default',bias='default',eta=0,iteration=1000,mini_batch_size=100):
        self.x = x
        # self.y = self.convert_label(y,sizes[-1])
        self.y = y
        self.sizes = sizes
        self.layers = len(sizes) - 1
        self.w_matrix = [np.random.randn(x,y) for x,y in zip(sizes[:-1],sizes[1:])] if weight == 'default' else weight
        self.b_matrix = [np.random.randn(y,1) for y in sizes[1:]] if bias == 'default' else bias
        self.eta = eta
        self.mini_batch_size = mini_batch_size
        self.iteration = iteration

    def convert_label(self,y,number_of_classes=3):
        Y = sparse.coo_matrix((np.ones_like(y),(y,np.arange(len(y)))),shape = (number_of_classes,len(y))).toarray()
        return Y

    @property
    def layers(self):
        return self.__layers

    @layers.setter
    def layers(self,layers):
        self.__layers = layers

    def cost(self,yhat,y):
        return -np.sum(y*np.log(yhat)) / len(y)

    def softmax(self,z):
        e_z = np.exp(z - np.max(z,axis=0,keepdims=True))
        z = e_z / e_z.sum(axis=0)
        return z

    def feed_forward(self, a):
        for i in range(self.layers):
            z = np.dot(self.w_matrix[i].T, a)+self.b_matrix[i]
            # if i == self.layers - 1:
            #     a = self.softmax(z)
            # else:
            #     a = np.maximum(z,0)
            a = sigmoid(z)
        return a

    #Single input_output
    def feed_forward_and_update(self,x_mini_batch,y_mini_batch,w_matrix,b_matrix):
        nabla_w = [np.zeros(w_matrix.shape) for w_matrix in w_matrix]
        nabla_b = [np.zeros(b_matrix.shape) for b_matrix in b_matrix]
        for input,output in zip(x_mini_batch.T,y_mini_batch.T):
            input = input.reshape(input.shape[0],1)
            output = output.reshape(output.shape[0],1)
            delta_nabla_w,delta_nabla_b = self.backprop(input,output,self.w_matrix,self.b_matrix)
            nabla_w = [nab_w + del_nab_w for nab_w,del_nab_w in zip(nabla_w,delta_nabla_w)]
            nabla_b = [nab_b + del_nab_b for nab_b,del_nab_b in zip(nabla_b,delta_nabla_b)]
            print(nabla_w)
        return nabla_w,nabla_b

    #From last output to first
    def backprop(self,x,y,w_matrix,b_matrix):
        a_vectors = [x]
        a_current = x
        z_vectors = []

        #feed-forward
        for i in range(self.layers):
            z_i = np.dot(w_matrix[i].T,a_current) + b_matrix[i]
            z_vectors.append(z_i)

            # if i == self.layers - 1:
            #     a_i = self.softmax(z_i)
            # else:
            #     a_i = np.maximum(z_i,0)
            a_i = sigmoid(z_i)

            a_vectors.append(a_i)
            a_current = a_i

        #back-propagation
        e = [(a_vectors.pop() - y)] #e_last
        delta_nabla_b = []
        delta_nabla_w = []
        e_next = 0
        for i in range(0,self.layers):
            a_vector = a_vectors[-(i+1)]
            z_vector = z_vectors[-(i+1)]
            # if i + 1 != 1:
            #     e[-1][z_vector <= 0] = 0 #reLU derivative
            # else:
            e[-1] = e[-1] * sigmoid_prime(z_vector)
            nabla_w_i = np.dot(a_vector,e[-1].T)
            nabla_b_i = e[-1]

            delta_nabla_w.insert(0,nabla_w_i)
            delta_nabla_b.insert(0,nabla_b_i)
            e_next = np.dot(w_matrix[-(i+1)],e[-1])
            e.append(e_next)
        return delta_nabla_w,delta_nabla_b

    def save_network(self,filename,location='#currentWorkDir#'):
        unallowed_characters = "/\\?%*:|\"<>"
        assert isinstance(filename,str) and len(filename) >= 1 and filename not in unallowed_characters,'-> invalid FILENAME to save...'
        assert isinstance(location,str) and len(location) >= 1 and location not in unallowed_characters,'-> invalid PATH to save...'
        if location != '#currentWorkDir#':
            assert os.path.exists(location),'-> PATH does not exist...'

        filename = filename.split(".")
        filename = ".".join(filename) if len(filename) == 2 else filename[0] + ".dat"
        path = filename if location == '#currentWorkDir#' else location + filename
        print('-> trying to save to %s' %(path))
        try:
            fh = open(path,'wb')
            pickle.dump((self.w_matrix,self.b_matrix),fh,pickle.HIGHEST_PROTOCOL)
            print('-> saved to \'%s\'...' %(path))
        except pickle.PicklingError:
            print('-> object is not pickle-able...')
            sys.exit(1)

    def load_network(self,path_to_filename):
        if not os.path.isfile(path_to_filename):
            print('-> invalid file path specified...')
            sys.exit(1)

        filename = path_to_filename
        try:
            fh = open(filename,'rb')
            w_matrix,b_matrix = pickle.load(fh)
            self.w_matrix = w_matrix
            self.b_matrix = b_matrix
            print('-> %s loaded to the network...' %(filename))
        except pickle.UnpicklingError:
            print('-> are you sure this is the correct saved-file ?...')
            sys.exit(1)


    def train(self,epoch=15,test_data = None):
        x_mini_batches = [self.x[:,k:k+self.mini_batch_size] for k in range(0,self.x.shape[1],self.mini_batch_size)]
        y_mini_batches = [self.y[:,k:k+self.mini_batch_size] for k in range(0,self.y.shape[1],self.mini_batch_size)]
        #feed-forward
        for epoc in range(epoch):
            print("-> epoch: ",epoc + 1)
            for x_mini_batch,y_mini_batch in zip(x_mini_batches,y_mini_batches):
                nabla_w,nabla_b = self.feed_forward_and_update(x_mini_batch,y_mini_batch,self.w_matrix,self.b_matrix)

                self.w_matrix = [w_matrix - (self.eta/len(x_mini_batch))*nabla_w_matrix for w_matrix,nabla_w_matrix in zip(self.w_matrix,nabla_w)]
                self.b_matrix = [b_matrix - (self.eta/len(x_mini_batch))*nabla_b_matrix for b_matrix,nabla_b_matrix in zip(self.b_matrix,nabla_b)]
            if test_data:
                print("-> epoch {} result: {} / {}".format(epoc + 1,self.evaluate(test_data),len(test_data[1])))
            else:
                print("-> epoch {} completed...")

    def evaluate(self,test_data):
        x_test , y_test = test_data
        test_results = []
        for x,y in zip(x_test.T,y_test):
            x = x.reshape(x.shape[0],1)
            y = y.reshape(y.shape[0],1)
            a = np.argmax(self.feed_forward(x))
            b = y
            test_results.append((a,b))
        return sum(int(x == y) for (x, y) in test_results)

if __name__ == '__main__':
    (x_train,y_train) , (x_test,y_test) = tf.keras.datasets.cifar10.load_data()

    x_train = x_train.reshape(x_train.shape[0],32*32*3).T
    y_train = y_train.T.reshape(y_train.shape[0])
    y_train = convert_onehot(y_train)

    x_test = x_test.reshape(x_test.shape[0],32*32*3).T
    sizes = [3072,100,10]
    network = DeepFeedForwardNetwork(x_train,y_train,sizes,eta=0.9,iteration=1)
    network.train(epoch=15,test_data = (x_test , y_test))
    network.save_network('network_save.dat')
