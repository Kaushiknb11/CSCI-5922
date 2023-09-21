#L = x**2 - 4*x
import numpy as np
# if exp
# for lambda -> 5*np.exp(5*x)

dL_dx = lambda x: 2*x - 4
dL_dx_str = "2({}) - 4"
# dL_dx = lambda x: 5*np.exp(5*x)
# dL_dx_str = "5(e^5*{})"
lr = .10

def compute_gradient(x,dL_dx):
    
    return dL_dx(x)

def display(x, gradient, dL_dx_str, lr, mode='initial'):
    
    if mode=='initial':
        
        print("When x is 0, the gradient is ",end='')
        print(dL_dx_str.format(x)+f" = {gradient}")
        print()
    
    else:

        for epoch in range(len(x) - 1):
            print(f"Update x (iteration {epoch + 1})")
            print(f"x_new = x_previous - LR * (gradient x_previous)")
            print(f"x_new = {np.round(x[epoch],2)} - ({lr})({np.round(gradient[epoch],2)}) = {np.round(x[epoch + 1],2)}")
            print("new gradient value: ", end="")
            print(dL_dx_str.format('x') + " -> ", end="")
            print(dL_dx_str.format(np.round(x[epoch + 1],2)) + f" = {np.round(gradient[epoch + 1],2)}")
            print()


        
    
    
    

    
def optimized(x_new,dL_dx):
    
    return dL_dx(x_new)

def epochs(iterations, x_new, gradient_prev, lr, dL_dx, prev_x, prev_grad):
    
    gradients_list = []
    all_xs = []
    
    all_xs.append(prev_x)
    gradients_list.append(prev_grad)
    
    iterations = iterations - 1
    for e in range(1,iterations):
        
        x_new = x_new - lr * gradient_prev
        gradient_prev = dL_dx(x_new)
        gradients_list.append(gradient_prev)
        all_xs.append(x_new)
        
#         print('Iteration: ',e)
#         print("X",x_new)
#         print("gradient",gradient_prev)
    



    display(all_xs, gradients_list, dL_dx_str, lr,mode='iterations')
        
        
    


x = 0
gradient_0 = dL_dx(x)
display(x, gradient_0, dL_dx_str, lr, mode='initial')

iterations = 7 # change this
iterations = iterations + 2

epochs(iterations, x, gradient_0, lr, dL_dx, x, gradient_0)