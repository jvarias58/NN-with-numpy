w1 = np.random.uniform(-0.5, 0.5, (80, 784)) 
b1 = np.random.uniform(-0.5, 0.5, (80, 1))
w2 = np.random.uniform(-0.5, 0.5, (40, 80))
b2 = np.random.uniform(-0.5, 0.5, (40, 1))
w3 = np.random.uniform(-0.5, 0.5, (10, 40))
b3 = np.random.uniform(-0.5, 0.5, (10, 1))
learn_rate = 0.01


def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def dsigmoid(z):
    return z*(1-z)


for epoch in range(20):
    for i in range(41000):
        
        x = x_train[:,i].reshape(784, 1)
        y = labels[:,i].reshape(10,1)

        #Forward propagation
        #First layer
        z1 = np.dot(w1, x) + b1 #(16 x 784) X (784 x 1) + (16 x 1) = (16 x 1)
        a1 = sigmoid(z1)
        
        #Second Layer
        z2 = np.dot(w2,a1) + b2 #(10 x 16) X (16 x 1) + (10 x 1) = (10 x 1)
        a2 = sigmoid(z2)
        
        z3 = np.dot(w3,a2) + b3 #(10 x 16) X (16 x 1) + (10 x 1) = (10 x 1)
        a3 = sigmoid(z3)
    
    
        #Backprop
        d3 = (a3 - y) * dsigmoid(a3)
        w3 += -learn_rate * np.dot(d3, np.transpose(a2))
        b3 += -learn_rate * d3
        
        #d2 = (a2 - y) * (a2 * (1 - a2))  
        d2 = np.dot(np.transpose(w3), d3) * dsigmoid(a2)
        w2 += -learn_rate * np.dot(d2, np.transpose(a1))
        b2 += -learn_rate * d2
       
        
        d1 = np.dot(np.transpose(w2), d2) * dsigmoid(a1)
        w1 += -learn_rate * np.dot(d1, np.transpose(x))
        b1 += -learn_rate * d1