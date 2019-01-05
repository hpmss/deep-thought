import numpy as np
import pickle
from scipy import sparse
import tensorflow as tf

def convert_onehot(values):
    a = np.zeros((values.size,values.max() + 1))
    a[np.arange(values.size),values] = 1
    return a.T

class DeepFeedForwardNetwork(object):

    def __init__(self,x,y,sizes,eta=0,iteration=1000,mini_batch_size=100):
        self.x = x
        # self.y = self.convert_label(y,sizes[-1])
        self.y = y
        self.sizes = sizes
        self.layers = len(sizes) - 1
        self.w_matrix = [np.random.randn(x,y) for x,y in zip(sizes[:-1],sizes[1:])]
        self.b_matrix = [np.random.randn(y,1) for y in sizes[1:]]
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

    #Single input_output
    def feed_forward(self,x_mini_batch,w_matrix,b_matrix):
        z_vectors = []
        a_vectors = [x_mini_batch]
        a_current = x_mini_batch #First feed-forward with input is x_mini_batch
        for i in range(self.layers):
            z_i = np.dot(w_matrix[i].T,a_current) + b_matrix[i]
            z_vectors.append(z_i)
            if i == self.layers - 1:
                a_i = self.softmax(z_i)
            else:
                a_i = np.maximum(z_i,0)

            a_vectors.append(a_i)
            a_current = a_i
        return z_vectors,a_vectors

    #From last output to first
    def backprop(self,z_vectors,a_vectors,w_matrix,b_matrix,y):
        e = [(a_vectors.pop() - y) / len(y)] #e_last
        nabla_w = []
        nabla_b = []
        e_next = 0
        for i in range(0,self.layers):
            a_vector = a_vectors[-(i+1)]
            z_vector = z_vectors[-(i+1)]
            if i + 1 != 1:
                e[-1][z_vector <= 0] = 0 #reLU derivative
            nabla_w_i = np.dot(a_vector,e[-1].T)
            nabla_b_i = e[-1]

            nabla_w.insert(0,nabla_w_i)
            nabla_b.insert(0,nabla_b_i)
            e_next = np.dot(w_matrix[-(i+1)],e[-1])
            e.append(e_next)
        return nabla_w,nabla_b

    def save_network(self,filename):
        unallowed_characters = "/\\?%*:|\"<>"
        assert isinstance(filename,str) and len(filename) >= 1 and filename not in unallowed_characters,'-> invalid filename to save...'
        filename = filename.split(".")
        filename = ".".join(filename) if len(filename) == 2 else filename[0] + ".dat"
        print(filename)
        return filename

    def train(self,epochs=15):
        x_mini_batches = [self.x[:,k:k+self.mini_batch_size] for k in range(0,self.x.shape[1],self.mini_batch_size)]
        y_mini_batches = [self.y[:,k:k+self.mini_batch_size] for k in range(0,self.y.shape[1],self.mini_batch_size)]
        #feed-forward
        for epoc in range(epochs):
            print("Epoch: ",epoc + 1)
            for x_mini_batch,y_mini_batch in zip(x_mini_batches,y_mini_batches):
                for i in range(self.iteration):
                    z_vectors , a_vectors = self.feed_forward(x_mini_batch,self.w_matrix,self.b_matrix)
                    if i % 100 == 0 and i >= 100:
                        loss = self.cost(a_vectors[-1],y_mini_batch)
                        print('iter %d,loss: %f' %(i,loss))
                    if i == 999:
                        z_vectors , a_vectors = self.feed_forward(x_mini_batch,self.w_matrix,self.b_matrix)
                        print(a_vectors[-1][:,:2])
                        print(y_mini_batch[:,:2])

                    nabla_w , nabla_b = self.backprop(z_vectors,a_vectors,self.w_matrix,self.b_matrix,y_mini_batch)
                    self.w_matrix = [w_matrix - self.eta*nabla_w for w_matrix,nabla_w in zip(self.w_matrix,nabla_w)]
                    self.b_matrix = [b_matrix - self.eta*nabla_b for b_matrix,nabla_b in zip(self.b_matrix,nabla_b)]
                z_vectors , a_vectors = self.feed_forward(x_mini_batch,self.w_matrix,self.b_matrix)
                loss = self.cost(a_vectors[-1],y_mini_batch)
                print('epoc %d,loss: %f' %(epoc + 1,loss))

        z_vectors , a_vectors = self.feed_forward(x_mini_batch,self.w_matrix,self.b_matrix)
        print(a_vectors[-1])

(x_train,y_train) , (x_test,y_test) = tf.keras.datasets.cifar10.load_data()

x_train = x_train.reshape(x_train.shape[0],32*32*3).T
y_train = y_train.T.reshape(y_train.shape[0])
y_train = convert_onehot(y_train)
sizes = [3072,100,10]
network = DeepFeedForwardNetwork(x_train,y_train,sizes,0.9,iteration=1000)
network.train(15)
network.save_network("network.psd")
