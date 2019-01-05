import numpy as np
import pickle
import os
import sys
from scipy import sparse
import tensorflow as tf

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

    def feedforward(self, a):
        for i in range(self.layers):
            z = np.dot(self.w_matrix[i].T, a)+self.b_matrix[i]
            if i == self.layers - 1:
                a = self.softmax(z)
            else:
                a = np.maximum(z,0)
        return a

    #Single input_output
    def feed_forward(self,x_mini_batch,w_matrix,b_matrix):
        all_output = []
        for input in x_mini_batch.T:
            input = input.reshape(input.shape[0],1)
            z_vectors = []
            a_vectors = [input]
            a_current = input
            for i in range(self.layers):
                z_i = np.dot(w_matrix[i].T,a_current) + b_matrix[i]
                z_vectors.append(z_i)
                if i == self.layers - 1:
                    a_i = self.softmax(z_i)
                else:
                    a_i = np.maximum(z_i,0)

                a_vectors.append(a_i)
                a_current = a_i
            all_output.append((z_vectors,a_vectors))

        return all_output

    #From last output to first
    def backprop(self,all_output,w_matrix,b_matrix,y):
        nabla_w = [np.zeros(w_matrix.shape) for w_matrix in w_matrix]
        nabla_b = [np.zeros(b_matrix.shape) for b_matrix in b_matrix]
        for (z_vectors,a_vectors),y_label in zip(all_output,y.T):
            y_label = y_label.reshape(y_label.shape[0],1)
            e = [(a_vectors.pop() - y_label)] #e_last
            delta_nabla_b = []
            delta_nabla_w = []
            e_next = 0
            for i in range(0,self.layers):
                a_vector = a_vectors[-(i+1)]
                z_vector = z_vectors[-(i+1)]
                if i + 1 != 1:
                    e[-1][z_vector <= 0] = 0 #reLU derivative
                nabla_w_i = np.dot(a_vector,e[-1].T)
                nabla_b_i = e[-1]

                delta_nabla_w.insert(0,nabla_w_i)
                delta_nabla_b.insert(0,nabla_b_i)
                e_next = np.dot(w_matrix[-(i+1)],e[-1])
                e.append(e_next)
            nabla_w = [nab_w + del_nab_w for nab_w,del_nab_w in zip(nabla_w,delta_nabla_w)]
            nabla_b = [nab_b + del_nab_b for nab_b,del_nab_b in zip(nabla_b,delta_nabla_b)]
        return nabla_w,nabla_b

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


    def train(self,epoch=15):
        x_mini_batches = [self.x[:,k:k+self.mini_batch_size] for k in range(0,self.x.shape[1],self.mini_batch_size)]
        y_mini_batches = [self.y[:,k:k+self.mini_batch_size] for k in range(0,self.y.shape[1],self.mini_batch_size)]
        #feed-forward
        for epoc in range(epoch):
            print("Epoch: ",epoc + 1)
            for x_mini_batch,y_mini_batch in zip(x_mini_batches,y_mini_batches):
                for i in range(self.iteration):
                    all_output = self.feed_forward(x_mini_batch,self.w_matrix,self.b_matrix)

                    nabla_w , nabla_b = self.backprop(all_output,self.w_matrix,self.b_matrix,y_mini_batch)
                    self.w_matrix = [w_matrix - (self.eta/self.mini_batch_size)*nabla_w_matrix for w_matrix,nabla_w_matrix in zip(self.w_matrix,nabla_w)]
                    self.b_matrix = [b_matrix - (self.eta/self.mini_batch_size)*nabla_b_matrix for b_matrix,nabla_b_matrix in zip(self.b_matrix,nabla_b)]
                # a_vector = self.feedforward(x_mini_batch)
                # loss = self.cost(a_vector,y_mini_batch)
                # print('epoc %d,loss: %f' %(epoc + 1,loss))


if __name__ == '__main__':
    (x_train,y_train) , (x_test,y_test) = tf.keras.datasets.cifar10.load_data()

    x_train = x_train.reshape(x_train.shape[0],32*32*3).T
    y_train = y_train.T.reshape(y_train.shape[0])
    y_train = convert_onehot(y_train)
    sizes = [3072,100,10]
    w_matrix = [np.ones((x,y)) for x,y in zip(sizes[:-1],sizes[1:])]
    b_matrix = [np.ones((y,1)) for y in sizes[1:]]
    network = DeepFeedForwardNetwork(x_train,y_train,sizes,eta=0.9,iteration=1)
    network.train(epoch=30)
    network.save_network('network_save.dat')
