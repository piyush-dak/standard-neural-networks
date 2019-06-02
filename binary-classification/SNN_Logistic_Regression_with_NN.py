
# coding: utf-8

# # Logistic Regression with a Neural Network mindset



import numpy as np
import matplotlib.pyplot as plt
import h5py #heirarchial data format
import scipy.misc as misc
from PIL import Image
from lr_utils import load_dataset


#%%
# 
# **Problem Statement**: 
#    You are given a dataset ("data.h5") containing:
#     - a training set of m_train images labeled as cat (y=1) or non-cat (y=0)
#     - a test set of m_test images labeled as cat or non-cat
#     - each image is of shape (num_px, num_px, 3) where 3 is for the 3 channels (RGB). Thus, each image is square (height = num_px) and (width = num_px).


# Loading the data (cat/non-cat)
x_train_set_image, y_train_set_label, x_test_set_image, y_test_set_label, classes = load_dataset()

# Example of a picture
index = 2
plt.imshow(x_train_set_image[index])
print ("y = " + str(y_train_set_label[:, index]) + ", it's a '" + 
       classes[np.squeeze(y_train_set_label[:, index])].decode("utf-8") +  "' picture.")

m_train = x_train_set_image.shape[0]
m_test = x_test_set_image.shape[0]
num_px = x_train_set_image.shape[1]


print ("Number of training examples: m_train = " + str(m_train))
print ("Number of testing examples: m_test = " + str(m_test))
print ("Height/Width of each image: num_px = " + str(num_px))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_set_x shape: " + str(x_train_set_image.shape))
print ("y_train_set_label shape: " + str(y_train_set_label.shape))
print ("test_set_x shape: " + str(x_test_set_image.shape))
print ("y_test_set_label shape: " + str(y_test_set_label.shape))



# Reshape the training and test examples

x_train_set_image_flatten = x_train_set_image.reshape(x_train_set_image.shape[0],-1).T
x_test_set_image_flatten = x_test_set_image.reshape(x_test_set_image.shape[0],-1).T

print ("x_train_set_image_flatten shape: " + str(x_train_set_image_flatten.shape))
print ("y_train_set_label shape: " + str(y_train_set_label.shape))
print ("x_test_set_image_flatten shape: " + str(x_test_set_image_flatten.shape))
print ("y_test_set_label shape: " + str(y_test_set_label.shape))
print ("sanity check after reshaping: " + str(x_train_set_image_flatten[0:5,0]))


#%% 
# Normalize dataset
train_set_x = x_train_set_image_flatten/255.
test_set_x = x_test_set_image_flatten/255.



# ## 4 - Building the parts of our algorithm ## 
# 1. Define the model structure (such as number of input features) 
# 2. Initialize the model's parameters
# 3. Loop:
#     - Calculate current loss (forward propagation)
#     - Calculate current gradient (backward propagation)
#     - Update parameters (gradient descent)

#%%
def sigmoid(z):
    """
    Arguments:
    z -- A scalar or numpy array of any size.
    Return:
    s -- sigmoid(z)
    """
    s = 1/(1+np.exp(-z))    
    return s

def initialize_with_zeros(dim):
    """
    Argument:
    dim -- size of the w vector we want (or number of parameters in this case)
    
    Returns:
    w -- initialized vector of shape (dim, 1)
    b -- initialized scalar (corresponds to the bias)
    """
    w = np.zeros([dim,1])
    b = 0
    
    return w, b

#%%
# Forward propagation

def propagate(w, b, X, Y):
    """
    Implement the cost function and its gradient for the propagation explained above

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

    Return:
    cost -- negative log-likelihood cost for logistic regression
    dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss with respect to b, thus same shape as b
    
    Tips:
    - Write your code step by step for the propagation. np.log(), np.dot()
    """
    
    m = X.shape[1]
    A = sigmoid(np.dot(w.transpose(),X)+b)                                    # compute activation
    cost = -1/m*np.sum(Y*np.log(A)+(1-Y)*np.log(1-A))                                 # compute cost
    dw = 1/m*np.dot(X,(A-Y).transpose())
    db = 1/m*np.sum((A-Y))

    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    
    grads = {"dw": dw,
             "db": db}
    
    return grads, cost


#%%
# ### 4.4 - Optimization

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    """
    This function optimizes w and b by running a gradient descent algorithm
    
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps
    
    Returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
    

    """
    
    costs = []
    
    for i in range(num_iterations):
    
        grads, cost = propagate(w,b,X,Y)        
        dw = grads["dw"]
        db = grads["db"]

        w = w-learning_rate*dw
        b = b-learning_rate*db
        
        if i % 100 == 0:
            costs.append(cost)
        
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs


#%%

def predict(w, b, X):
    '''
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
    
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    
    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    '''
    
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)

    A = sigmoid(np.dot(w.transpose(),X)+b)
    
    for i in range(A.shape[1]):
        if A[0,i]<=0.5:
            Y_prediction[0,i]=0
        else:
            Y_prediction[0,i]=1    
    
    assert(Y_prediction.shape == (1, m))
    
    return Y_prediction

#%%
def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    """
    Builds the logistic regression model by calling the function you've implemented previously
    
    Arguments:
    X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
    Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
    X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
    Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
    num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
    print_cost -- Set to true to print the cost every 100 iterations
    
    Returns:
    d -- dictionary containing information about the model.
    """
    

    w, b = initialize_with_zeros(X_train.shape[0])
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    w = parameters["w"]
    b = parameters["b"]
    
    # Predict test/train set examples (≈ 2 lines of code)
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)


    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d

#%%

d = model(train_set_x, y_train_set_label, test_set_x, y_test_set_label, num_iterations = 2000, learning_rate = 0.005, print_cost = True)

#%%
# Example of a picture that was wrongly classified.
index = 10
plt.imshow(test_set_x[:,index].reshape((num_px, num_px, 3)))
print ("y = " + str(y_test_set_label[0,index]) + ", you predicted that it is a \"" + 
       classes[int(d["Y_prediction_test"][0,index])].decode("utf-8") +  "\" picture.")

# Let's also plot the cost function and the gradients.

# Plot learning curve (with costs)
costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()

#%%

learning_rates = [0.01, 0.001, 0.0001]
models = {}
for i in learning_rates:
    print ("learning rate is: " + str(i))
    models[str(i)] = model(train_set_x, y_train_set_label, test_set_x, y_test_set_label, num_iterations = 1500, learning_rate = i, print_cost = False)
    print ('\n' + "-------------------------------------------------------" + '\n')

for i in learning_rates:
    plt.plot(np.squeeze(models[str(i)]["costs"]), label= str(models[str(i)]["learning_rate"]))

plt.ylabel('cost')
plt.xlabel('iterations (hundreds)')

legend = plt.legend(loc='upper center', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()

#%%

my_image = "cat.jpg"   # change this to the name of your image file 

fname = "images/" + my_image
image = np.array(plt.imread(fname))
# Preprocess Image
my_image = misc.imresize(image, size=(num_px,num_px)).reshape((1, num_px*num_px*3)).T
my_predicted_image = predict(d["w"], d["b"], my_image)

plt.imshow(image)
print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + 
      classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")

# Bibliography:
# - http://www.wildml.com/2015/09/implementing-a-neural-network-from-scratch/
# - https://stats.stackexchange.com/questions/211436/why-do-we-normalize-images-by-subtracting-the-datasets-image-mean-and-not-the-c
