import numpy as np
from scipy import sparse

class RecurrentNeuralNetwork(object):

    def __init__(self,x,y,sizes,eta=0,iteration=1000,mini_batch_size=100):
        self.x = x
        self.y = self.convert_label(y,sizes[-1])
        self.mini_batch_size = mini_batch_size
        self.sizes = sizes
        self.layers = len(sizes) - 1
        self.w_matrix = [np.random.randn(x,y) for x,y in zip(sizes[:-1],sizes[1:])]
        self.b_matrix = [np.random.randn(y,1) for y in sizes[1:]]
        self.h_matrix = [np.random.randn(x,x) for x in sizes[1:-1]]
        self.h_previous_init = [np.zeros((self.w_matrix[i].shape[1],self.mini_batch_size)) for i in range(self.layers - 1)]
        self.eta = eta
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
        return -np.sum(y*np.log(yhat)) / y.shape[1]

    def softmax(self,z):
        e_z = np.exp(z - np.max(z,axis=0,keepdims=True))
        z = e_z / e_z.sum(axis=0)
        return z

    def tanh(self,x):
        return (np.exp(2*x) - 1)/(np.exp(2*x) + 1)

    def tanh_derivative(self,x):
        return 1 - self.tanh(x) * self.tanh(x)

    #Single input_output
    #[2,100,3] (100x2).(2x100) -> 100x100
    def feed_forward(self,x_mini_batch,w_matrix,b_matrix,h_matrix):
        z_vectors = []
        a_vectors = [x_mini_batch]
        a_current = x_mini_batch #First feed-forward with input is x_mini_batch
        for i in range(self.layers):
            if i != (self.layers - 1):
                z_i = np.dot(h_matrix[i],self.h_previous_init[i]) + (np.dot(w_matrix[i].T,a_current) + b_matrix[i])
                z_vectors.append(z_i)
                a_i = self.tanh(z_i)
                self.h_previous_init[i] = a_i
                a_vectors.append(a_i)
                a_current = a_i
            else:
                z_last = np.dot(w_matrix[i].T,a_current) + b_matrix[i]
                z_vectors.append(z_last)
                a_last = self.softmax(z_last)
                a_vectors.append(a_last)

        return z_vectors,a_vectors

    #From last output to first
    def backprop(self,z_vectors,a_vectors,w_matrix,b_matrix,y):
        e = [(a_vectors.pop() - y) / len(y)] #e_last

        nabla_w = []
        nabla_b = []
        nabla_h =  []
        e_next = 0
        for i in range(0,self.layers):
            a_vector = a_vectors[-(i+1)]
            z_vector = z_vectors[-(i+1)]
            nabla_w_i = np.dot(a_vector,e[-1].T)
            nabla_b_i = np.sum(e[-1],axis=1,keepdims=True)
            if 0 < i < self.layers :
                nabla_h_i = np.dot(e[-1],self.h_previous_init[i-1])
                nabla_h.insert(0,nabla_h_i)

            nabla_w.insert(0,nabla_w_i)
            nabla_b.insert(0,nabla_b_i)
            if -(i + 1) != -1:
                e_next = e_next * self.tanh_derivative(z_vector)
            e_next = np.dot(w_matrix[-(i+1)],e[-1])
            e.append(e_next)
        return nabla_w,nabla_b,nabla_h

    def train_rnn(self):
        x_mini_batches = [self.x[:,k:k+self.mini_batch_size] for k in range(0,self.x.shape[1],self.mini_batch_size)]
        y_mini_batches = [self.y[:,k:k+self.mini_batch_size] for k in range(0,self.y.shape[1],self.mini_batch_size)]
        #feed-forward
        random_batch_pickout = np.random.permutation(len(x_mini_batches))
        for j in random_batch_pickout:
            y_mini_batch = y_mini_batches[j]
            x_mini_batch = x_mini_batches[j]
            print('Batch id: ',j)
            for i in range(self.iteration):
                z_vectors , a_vectors = self.feed_forward(x_mini_batch,self.w_matrix,self.b_matrix,self.h_matrix)
                if i % 100 == 0:
                    loss = self.cost(a_vectors[-1],y_mini_batch)
                    print('iter %d,loss: %f' %(i,loss))

                nabla_w , nabla_b,nabla_h = self.backprop(z_vectors,a_vectors,self.w_matrix,self.b_matrix,y_mini_batch)
                self.w_matrix = [w_matrix - self.eta*nabla_w for w_matrix,nabla_w in zip(self.w_matrix,nabla_w)]
                self.b_matrix = [b_matrix - self.eta*nabla_b for b_matrix,nabla_b in zip(self.b_matrix,nabla_b)]
                self.h_matrix = [h_matrix - self.eta*nabla_h for h_matrix,nabla_h in zip(self.h_matrix,nabla_h)]
        z_vectors , a_vectors = self.feed_forward(x_mini_batch,self.w_matrix,self.b_matrix,self.h_matrix)
        print(a_vectors[-1])


sizes = [2,100,3]
N = 100
y = np.random.permutation(sizes[-1])
y_copy = y.copy()
for i in range(N - 1):
    y = np.concatenate((y,y_copy),axis=0)
print(y)
y = np.zeros((sizes[-1] * N))
# y = np.random.randint(0,3,(sizes[-1] * N),dtype='uint8')
X = np.random.randn(sizes[0],N*sizes[-1])

eta = 0.05
network = RecurrentNeuralNetwork(X,y,sizes,eta,iteration=1000)
network.train_rnn()
