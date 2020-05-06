import numpy as np
from time import time
import matplotlib.pyplot as plt

def feature_scaling(x):
    one_over_r = [[1/(max(x[i]) - min(x[i]))] if max(x[i]) - min(x[i]) != 0 else [1] for i in range(len(x))]
    alpha = [[min(x[i])] if max(x[i]) - min(x[i]) != 0 else [0] for i in range(len(x))]
    x_prime = np.multiply((np.matrix(x)-np.array(alpha)),np.array(one_over_r))
    return(x_prime,alpha,one_over_r)

class linear_regression:
            
    def __init__(self,data,result,variable_names,hypothesis_equation):
    
        self.initial_data = data
        self.result = result
        self.variable_names = variable_names
        self.hypothesis_equation = hypothesis_equation
        
        # we apply f_i to each data point
        data_prime = [[] for i in range(len(hypothesis_equation))]
        for i in range(len(data[0])):
            for j in range(len(variable_names)):
                exec('%s = %.7f' % (variable_names[j],data[j][i]))
            for n in range(len(hypothesis_equation)):
                data_prime[n].append(eval(hypothesis_equation[n]))
        
        # we feature scaling here
        
        self.analytical_data = data_prime
        self.data,self.alpha_list,self.one_over_r = feature_scaling(data_prime)
        self.data_prime = self.data
        self.list_of_obsrervations = self.data.transpose()
    
    def cost_function(self,thita_list):
        total = 0 
        for i in range(self.list_of_obsrervations.shape[0]):
            total += pow(int(np.dot(self.list_of_obsrervations[i],np.array(thita_list))) - self.result[i],2)
        return(total/(2*len(self.result)))
    
    def partial_derivative(self,thita_list,k):
        total = 0 
        for i in range(self.list_of_obsrervations.shape[0]):
            total += (int(np.dot(self.list_of_obsrervations[i],np.array(thita_list))) - self.result[i])*self.list_of_obsrervations.item(i,k)
        return(total/len(self.result))
        
    def gradient_descent(self,timeout_error = 60,alpha=0.15,initial_guess='Default',print_message=True):
        
        start = time()
        now = time()
        current_thita_list = initial_guess if initial_guess != 'Default' else [0 for _ in range(len(self.hypothesis_equation))]
        
        
        repetition = 0
        ax = plt.subplot(111)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.xlabel('Number of trials')
        plt.ylabel('Cost')
        plt.suptitle('Cost as Number of Trials Increase')
        
        try:        
            while now - start < timeout_error:
            
                partial_derivatives = [self.partial_derivative(current_thita_list,i) for i in range(len(current_thita_list))]

                if [round(i,5) for i in partial_derivatives] == [float(0) for i in range(len(partial_derivatives))]:
                  return([round(j,3) for j in current_thita_list])
                  break

                current_thita_list =[current_thita_list[i] - alpha*partial_derivatives[i] for i in range(len(partial_derivatives))]
                    
                now = time()
                repetition +=1
                
                geoant = self.cost_function(current_thita_list)
                plt.title(f'Run number {repetition} with cost {round(geoant,2)}')
                plt.scatter(repetition,geoant,color='black')
                plt.pause(0.05)
            plt.show()
        except KeyboardInterrupt:
            pass
            
        self.theta = [round(j,3) for j in current_thita_list]
        self.method = 'Numerical'
        if print_message:
            print('Your model has been trained!')
            
        
    def predict(self,value):
        # use the same like above to find the equation
        if not hasattr(self, 'theta'):
            print('Your model has not been trained!')
        else:
            if self.method == 'Numerical':
                equation = ''
                for i in range(len(self.hypothesis_equation)):
                    if i != 0:
                        equation += '+'
                    equation += f'{self.theta[i] * self.one_over_r[i][0]} * ({self.hypothesis_equation[i]} - {self.alpha_list[i][0]})'
                    
                for j in range(len(self.variable_names)):
                    exec('%s = %.7f' % (self.variable_names[j],value[j]))
                
                return(eval(equation))
                
            elif self.method == 'Analytic':
                equation = ''
                for i in range(len(self.hypothesis_equation)):
                    if i != 0:
                        equation += '+'
                    equation += f'{self.theta[i]} * ({self.hypothesis_equation[i]})'
                    
                for j in range(len(self.variable_names)):
                    exec('%s = %.7f' % (self.variable_names[j],value[j]))
                    
                return(eval(equations))    
            
    def analytic(self):
        data = np.matrix(self.analytical_data) 
        a = np.dot(np.dot(np.linalg.inv(np.dot(data,data.transpose())),data),self.result).tolist()[0]
        self.theta = a
        self.method = 'Analytic'
        return(a)
        
    def output_result(self,precision = 2):
        if not hasattr(self, 'theta'):
            print('Your model has not been trained!')
        else:
            if self.method == 'Numerical':
                for i in range(len(self.hypothesis_equation)):
                    if i != 0:
                        print(' + ',end = '')
                    print(f'{round(self.theta[i] * self.one_over_r[i][0],precision)} * ({self.hypothesis_equation[i]} - {round(self.alpha_list[i][0],precision)}',end= '')
                print('')
            elif self.method == 'Analytic':
                for i in range(len(self.hypothesis_equation)):
                    if i != 0:
                        print(' + ',end = '')
                    print(f'{round(self.theta[i],precision)} * {self.hypothesis_equation[i]}',end= '')
                print('')