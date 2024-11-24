"""
Author   : Shao Hsuan Hung, (s.hung@student.tue.nl)
Objective: A multi-layer-preceptron object from scratch, and in the __main__ function, build a simple DL training pipline for demonstration. 
Remark   : If you want to run the main function, you should have the mnist dataset in the current directory
Python: 3.8.15
"""
import numpy as np
class MLP(object):
    def __init__(self,layer_dimension,layer_act_fun):
        '''
        :parameters
        layer_dimension: list of dimension of each layer, including the input/output layer, e.g.: [784,128,64,32,10]
        layer_act_fun: list of string of name of active function, e.g:[None,"ReLU","ReLU","ReLU","Softmax"]
        (Only implement: ReLU, Softmax and Sigmoid, cross entropy function)
        '''
        self.layer_dimension = list(layer_dimension)
        self.layer_act_fun = list(layer_act_fun)
        self.num_layer = len(self.layer_dimension) #Including Input + hidden + output
        # Initialize the weight and bias, nueron and nueron before activation
        self.W = [0]*(self.num_layer)
        self.B = [0]*(self.num_layer)
        self.Z = [0]*(self.num_layer)
        self.A = [0]*(self.num_layer)
        # derivate of weighs, bias, and the output layer, for the back propogation 
        self.dW = [0]*(self.num_layer)
        self.dB = [0]*(self.num_layer)
        self.dA_output = [0]*(self.num_layer)
        # Random assign the inital weigh and bias
        for layer_idx in range(1,self.num_layer):
            self.W[layer_idx] = np.random.randn(self.layer_dimension[layer_idx],self.layer_dimension[layer_idx-1])*0.01
            self.B[layer_idx] = np.zeros((self.layer_dimension[layer_idx],1))
        # Data log
        self.train_acc = []
        self.valid_acc = [] 
        self.loss = []
    def forward_propogation(self,x):
        self.A[0] = x
        for layer_idx in range(1,self.num_layer):
            self.Z[layer_idx],self.A[layer_idx] =  self.layer_forward_calculation(self.A[layer_idx-1],self.W[layer_idx],self.B[layer_idx],self.layer_act_fun[layer_idx])

        output = self.A[self.num_layer-1]
        return output

    def back_propogation(self):
        for i in reversed(range(1,self.num_layer)):
            self.dA_output[i-1],self.dW[i],self.dB[i] = self.layer_backward_calculation(self.W[i],self.B[i],self.Z[i],self.A[i-1],self.A[i],self.dA_output[i],self.layer_act_fun[i])
    
    def train(self,X,Y,X_valid,Y_valid,epochs,lr,mini_batch_size):
        m = X.shape[1]
        for one_epoch in range(epochs):
            cost = 0
            # generate the new mini-batch every epoch
            mini_batch_idx_list = self.mini_batch_shuffle(m,mini_batch_size)# generate the index list for x and y
            for idx in range(0,len(mini_batch_idx_list)):
                X_mini = X[:,mini_batch_idx_list[idx]]
                Y_mini = Y[:,mini_batch_idx_list[idx]]
                self.forward_propogation(X_mini)
                loss = self.loss_function(Y_mini)
                self.back_propogation()
                self.update_parameters(lr)
                cost = cost+loss
            # Do the inference to estimate the accuracy
            self.train_acc.append(self.acc_calculation(X,np.argmax(Y,axis=0).reshape((1,-1))))
            self.valid_acc.append(self.acc_calculation(X_valid,Y_valid))
            self.loss.append(cost/len(mini_batch_idx_list)) # Average loss
            # In end of the epoch:
            if((one_epoch+1)%10 == 0):
                print("-"*80)
                print("Epoch {}, Train loss:{:.4f}, Train Acc:{:2f}%, Valid Acc:{:2f}%".format(one_epoch+1,\
                                                                                    cost/len(mini_batch_idx_list),\
                                                                                    100*self.train_acc[one_epoch],\
                                                                                    100*self.valid_acc[one_epoch]))

    def acc_calculation(self,X_valid,Y_valid_label):
        Y_valid_predict = self.forward_propogation(X_valid)
        # Onehot encoding again
        Y_valid_predict = np.argmax(Y_valid_predict,axis=0).reshape(1,-1)
        #Y_valid_predict[Y_valid_predict<=0.5] = 0# max is 1, other is 0
        return np.sum(Y_valid_predict==Y_valid_label)/Y_valid_label.shape[1]
    def mini_batch_shuffle(self,m,mini_batch_size):
            random_idx = np.random.permutation(m)
            num_mini_batch = int(np.floor(m/mini_batch_size))
            random_idx_list = []
            for idx in range(0,num_mini_batch):
                random_idx_list.append(random_idx[(idx)*mini_batch_size:(idx+1)*mini_batch_size])
            if num_mini_batch < (m/mini_batch_size): # some index are not assign in the list, then add the rest
                random_idx_list.append(random_idx[num_mini_batch*mini_batch_size:])
            return random_idx_list
    def loss_function(self,Y):
        output = self.A[self.num_layer-1] # Get the output's value
         # Calculate the cross entropy loss
        if self.layer_act_fun[self.num_layer-1]=="Softmax":
            loss = -np.sum(np.sum(Y*np.log(output+1e-8),keepdims=True))/(Y.shape[0])
            dA = Y/(output+1e-8)# calcualte the dA for backpropogation

        else: # logistic cost
            loss = -np.sum(Y*np.log(output+1e-8)+(1-Y)*np.log(1-output))/Y.shape[0]
            dA = (-(Y/output)+((1-Y)/(1-output)))
        self.dA_output[self.num_layer-1] = dA
        return loss
    def update_parameters(self,lr):
        for layer in range(1,self.num_layer):
            self.W[layer] = self.W[layer]-lr*self.dW[layer]
            self.B[layer] = self.B[layer]-lr*self.dB[layer]

    def layer_forward_calculation(self,x,w,b,acti_fun):
        '''
        Parameter:
        x: (1) input nueron of  input layer, (2) A, nuerons in hidden layer
        w: weights of that layer
        b: biases of that layer Y

        Retrun:
        Z: New value of that nueron before activation (need to use it in back propogation)
        A: New value of that nueron after activation
        '''
        Z = np.dot(w,x)+b
        if acti_fun == "ReLU":
            A = self.ReLU(Z)
        elif acti_fun == "Sigmoid":
            A = self.sigmoid(Z)
        elif acti_fun == "Softmax":
            A = self.softmax(Z)
        else:
            A = self.softmax(Z)
        return Z,A
    def layer_backward_calculation(self,w,b,z,A_pervious,A,dA,activation):
        '''
        parameter:
        Retrun:
        '''
        if activation=="ReLU":
            G = self.ReLU(z,df=True)
            dZ = dA*G

        elif activation=="Sigmoid":
            G = self.sigmoid(z,df=True)
            dZ = dA*G

        elif activation=="Softmax":
            dZ = A*(1-dA)
        else:
            print("None")
        
        dW = np.dot(dZ,A_pervious.transpose())/z.shape[0]
        dB = np.sum(dZ,keepdims=True)/z.shape[0]
        dA_previous = np.dot(w.transpose(),dZ)
        return dA_previous,dW,dB

    @staticmethod
    def ReLU(z,df = False):
        if df == False:
            return (0 < z)*z
        else:
            return (0<z)*1
    @staticmethod
    def sigmoid(z,df = False):
        f = 1/(1+np.exp(-z)+1e-8) # add 1e-8 to avoid overflow
        if df == False:
            return f
        else:
            return f*(1-f)
    @staticmethod
    def softmax(z,df = False):
        if df == False:
            return np.exp(z)/(np.sum(np.exp(z),axis=0,keepdims=True))

    def save(self, filename):
        import json # using jason to save the training data only
        """Save the neural network to the file ``filename``."""
        data = {"Dimensions": self.layer_dimension,
                "Activations":self.layer_act_fun,
                "weights": [np.array(w).tolist() for w in self.W],
                "biases": [np.array(b).tolist() for b in self.B],
                "Z":[np.array(z).tolist() for z in self.Z],
                "A":[np.array(A).tolist() for A in self.A],
                "der. weights":[np.array(dw).tolist() for dw in self.dW],
                "der. biases":[np.array(db).tolist() for db in self.dB],
                "der. A":[np.array(dA).tolist() for dA in self.dA_output],
                }
        f = open(filename, "w")
        json.dump(data, f)
        f.close()
    def data_log(self,filename):
        import json
        data = {"loss":self.loss,
                "Train Acc":self.train_acc,
                "Valid Acc":self.valid_acc,
        }
        f = open(filename,"w")
        json.dump(data,f)
        f.close()
    def load_data(self,filname):
        import json
        f = open(filname,"r")
        data = json.load(f)
        self.layer_dimension = data["Dimensions"]
        self.layer_act_fun = data["Activations"]
        self.W = [np.array(w) for w in data["weights"]]
        self.B = [np.array(b) for b in data["biases"]]
        self.Z = [np.array(z) for z in data["Z"]]
        self.A = [np.array(a) for a in data["A"]]
        self.dW = [np.array(dw) for dw in data["der. weights"]]
        self.dB = [np.array(db) for db in data["der. biases"]]
        self.dA_output = [np.array(dA) for dA in data["der. A"]]

def one_hot_coding(y):
    classes = np.reshape(np.unique(y),(-1,1))
    Y_enc = np.equal(classes,y)*1
    return Y_enc

if __name__ == "__main__":
    print("The training result display for every 10 epoch.\n")
    ################ Please specify directory of your dataset here ######################
    TRAIN_DATA = './dataset/mnist_train.csv'
    TEST_DATA = './dataset/mnist_test.csv'
    #####################################################################################
    # Prepare the data set
    import pandas as pd
    train = np.array(pd.read_csv(TRAIN_DATA))
    test = np.array(pd.read_csv(TEST_DATA))
    train_img = train[:,1:].transpose()
    train_img = train_img/255
    train_label = np.reshape(train[:,0],(1,-1))
    train_label = one_hot_coding(train_label)
    valid_img = test[:,1:].transpose()
    valid_img = valid_img/255
    valid_label = np.reshape(test[:,0],(1,-1))

    # Build and train the model
    layer_dim = [784,128,64,32,10]
    avti=[None,"ReLU","ReLU","ReLU","Softmax"]
    network = MLP(layer_dim,avti)
    network.train(train_img,train_label,valid_img,valid_label,epochs=200,lr=0.01,mini_batch_size=32)
    
    # Save the checkpoint and performance datalog
    network.save("0109_train200_3ReLU_parameters")
    network.data_log("0109_train200_2ReLU_datalog")

    # Example for build a checkpoint
    # network.save({FILENAME})
    # Example for load the checkpoint,
    # First, create a new MLP instance, e.g.: new_network = MLP([784,200,100,10],[None,"ReLU","ReLU","Softmax"])
    # Second, using the helper function in the MLP object. e.g.: new_network.load_data({CHECKPOINT FILENAME})
    