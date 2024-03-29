# -*- coding: utf-8 -*-
"""
Special Thanks to Shaoyu Chen for helping with the code!
"""
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
  	return 1 / (1 + np.exp(-x))

def bp(I,D,n_hidden,eta,n_max):
    np.random.seed(1926)
    
    r_inp, c_inp = I.shape
    n_examples = r_inp;	
    n_input    = c_inp
    n_hidden   = n_hidden
    r_out,c_out = D.shape
    n_output   = c_out

    w = np.random.random((n_input, n_hidden))
    wb = np.random.random(n_hidden)
    v = np.random.random((n_hidden, n_output))
    vb = np.random.random(n_output)
    err_curve = np.zeros((n_max,c_out))

    for n in range(n_max):
        sq_err_sum = np.zeros((1,n_output))
    
        for k in range(n_examples):
            x = I[k,:].reshape([1,-1])
            z = sigmoid(x.dot(w)+wb)
            y = sigmoid(z.dot(v)+vb)

            # For (i)
            #if n == 4999:
            #    print("y%d = %f"%(k+1, y))


            err = (D[k,:] - y).reshape([1,-1])
            sq_err_sum += 0.5*np.square(err)
            
            Delta_output = err*(1-y)*y    
            Delta_v = z.T.dot(Delta_output)
            Delta_vb = np.sum(Delta_output,axis=0)

            Delta_hidden = Delta_output.dot(v.T)*(1-z)*z
            Delta_w = x.T.dot(Delta_hidden)    # extra code
            Delta_wb = np.sum(Delta_hidden,axis=0) # extra code
        
            v += eta*Delta_v
            vb += eta*Delta_vb
            w += eta*Delta_w
            wb += eta*Delta_wb

            err_curve[n] = sq_err_sum/n_examples

        print('epoch %d: err %f'%(n,np.mean(sq_err_sum)/n_examples))

        #For (ii)
        if n == 9999:
            input = np.array([1, 0, 1, 0, 1, 0, 1, 0]).reshape([1,-1])
            newz = sigmoid(input.dot(w)+wb)
            newy = sigmoid(newz.dot(v)+vb)
            print(newy)

    plt.plot(np.linspace(0,n_max-1,n_max),np.mean(err_curve,axis=1))
    #plt.show()
    return w,wb,v,vb,err_curve     


#Examples:
#1. XOR
#w,wb,v,vb,err_curve = bp(np.array([[0,0],[0,1],[1,0],[1,1]]), np.array([0,1,1,0]).reshape([-1,1]), 2, 4.5, 5000);
#    
#2. autoassociation
w,wb,v,vb,err_curve = bp(np.eye(8,8), np.eye(8,8), 3, 5.0, 10000)
#    
#3. function approximation
#x = np.arange(-np.pi,np.pi,0.1)
#fx = (np.cos(x)+1)/2
#fx = np.cos(x)
#w,wb,v,vb,err_curve = bp((x/np.pi).reshape([-1,1]),fx.reshape([-1,1]), 4, 2.0,4000);