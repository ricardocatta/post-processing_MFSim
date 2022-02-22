import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import statistical_module as sm



def absolute_error(a, b):
    a1 = np.sqrt((a**2).sum())
    b1 = np.sqrt((b**2).sum())
    abs_error = abs(a1 - b1)
    return abs_error

def relative_error(a, b):
    a1 = np.sqrt((a**2).sum())
    return (absolute_error(a, b)/a1)* 100

"""
Plot for mean w
"""
def plot_mean_w(dataframe, exp_w_mean, dataframe3, dataframe4, dataframe5, n):

    x = dataframe.x_comp * (1.0 /0.15)
    y = dataframe.w_average
    
    x_exp = exp_w_mean.x1_exp
    y_exp = exp_w_mean.mean_w_exp

    dif_mymodel = y - y_exp
    L2_norm_mymodel = np.sqrt((dif_mymodel ** 2).sum()) / len(np.array(y))

    #dif_article = y_exp1 - y_exp
    #L2_norm_article = np.sqrt((dif_article ** 2).sum()) / len(np.array(y))

    #print("Norma L2 de w_average Cs=0.1 : {0:.3f}% " .format(L2_norm_mymodel * 100))
#
    #print("Norma L2 de w_average Cs=0.2 : {0:.3f}% " .format(L2_norm_article * 100))
#
    #print("Norma Loo de w_average Cs=0.1 : {0:.3f}% " .format(np.amax(abs(dif_mymodel.sum()))))

    #print("Norma Loo de w_average Cs=0.2 : {0:.3f}% " .format(np.amax(abs(dif_article.sum()))))

    plt.style.use('ggplot')

    fig = plt.figure(figsize=(6.0, 4.0))
    axes1 = fig.add_subplot(1, 1, 1)
    axes1.set_ylabel('average_w')
    axes1.set_xlabel('x / L')
    
    if n == 3:
        plt.plot(x, y, '-', label="Cs = 0.1")
        plt.plot(x_exp, y_exp, '*', color="green", label="Experimental")
        x_exp3 = dataframe3.x_comp * (1.0 /0.15)
        y_exp3 = dataframe3.w_average 
        plt.plot(x_exp3, y_exp3, '--', color="blue", label="malha fixa")
        plt.legend(loc='best')
        fig.tight_layout()
        plt.show()
    elif n == 4:
        plt.plot(x, y, '-', label="van driest")
        plt.plot(x_exp, y_exp, '*', color="green", label="Experimental")
        x_exp3 = dataframe3.x_comp * (1.0 /0.15)
        y_exp3 = dataframe3.w_average 
        plt.plot(x_exp3, y_exp3, '--', color="blue", label="malha fixa")
        x_exp4 = dataframe4.x_comp * (1.0 /0.15)
        y_exp4 = dataframe4.w_average 
        plt.plot(x_exp4, y_exp4, '--', color="purple", label="malha fixa")
        plt.legend(loc='best')
        fig.tight_layout()
        plt.show()
    elif n == 5:
        plt.plot(x, y, '-', label="van driest")
        plt.plot(x_exp, y_exp, '*', color="green", label="Experimental")
        x_exp3 = dataframe3.x_comp * (1.0 /0.15)
        y_exp3 = dataframe3.w_average 
        plt.plot(x_exp3, y_exp3, '--', color="blue", label="malha fixa")
        x_exp4 = dataframe4.x_comp * (1.0 /0.15)
        y_exp4 = dataframe4.w_average 
        plt.plot(x_exp4, y_exp4, '--', color="purple", label="malha fixa")
        x_exp5 = dataframe5.x_comp * (1.0 /0.15)
        y_exp5 = dataframe5.w_average 
        plt.plot(x_exp5, y_exp5, '--', color="yellow", label="malha fixa")
        plt.legend(loc='best')
        fig.tight_layout()
        plt.show()
    else:
        plt.plot(x, y, '-', label="Van Driest Cs = 0.2")
        plt.plot(x_exp, y_exp, '*', color="green", label="Experimental")
        plt.legend(loc='best')
        fig.tight_layout()
        plt.show()

    
    
    
        
    #plt.plot(x_exp1, y_exp1, '--', color="blue", label="malha fixa")
    #plt.legend(loc='best')
    #fig.tight_layout()
    #plt.show()

    #print("Erro relativo Cs=0.2: {0:.3f}% " .format(relative_error(y_exp, y_exp1)))
    #print("Erro relativo Cs=0.1 {0:.3f}% " .format(relative_error(y_exp, y)))
    #print("----------------------------------")
    #print("Erro absoluto Cs=0.2: ", absolute_error(y_exp, y_exp1))
    #print("Erro absoluto Cs=0.1: ", absolute_error(y_exp, y))


"""
Plot for std w
"""
def plot_std_w(dataframe, exp_w_std, dataframe3, dataframe4, dataframe5, n):
    
    x = dataframe.x_comp * (1.0 /0.15)
    y = dataframe.w_stddev

    x_exp = exp_w_std.x1_exp
    y_exp = exp_w_std.std_w_exp
    
    #x_exp1 = experimental1.x_comp * (1.0 /0.15)
    #y_exp1 = experimental1.w_stddev

    dif_mymodel = y - y_exp
    L2_norm_mymodel = np.sqrt((dif_mymodel ** 2).sum()) / len(np.array(y))

    #dif_article = y_exp1 - y_exp
    #L2_norm_article = np.sqrt((dif_article ** 2).sum()) / len(np.array(y))
#
    #print("Norma L2 de w_std Cs=0.1 : {0:.3f}% " .format(L2_norm_mymodel * 100))
    #print("Norma L2 de w_std Cs=0.2 : {0:.3f}% " .format(L2_norm_article * 100))
    #print("Norma Loo de w_std Cs=0.1 : {0:.3f}% " .format(np.amax(abs(dif_mymodel.sum()))))
    #print("Norma Loo de w_std Cs=0.2 : {0:.3f}% " .format(np.amax(abs(dif_article.sum()))))

    

    plt.style.use('ggplot')

    fig = plt.figure(figsize=(6.0, 4.0))
    axes1 = fig.add_subplot(1, 1, 1)
    axes1.set_ylabel('std_w')
    axes1.set_xlabel('x / L')
 
    
    if n == 3:
        plt.plot(x, y, '-', label="van driest")
        plt.plot(x_exp, y_exp, '*', color="green", label="Experimental")
        x_exp3 = dataframe3.x_comp * (1.0 /0.15)
        y_exp3 = dataframe3.w_stddev 
        plt.plot(x_exp3, y_exp3, '--', color="blue", label="malha fixa")
        plt.legend(loc='best')
        fig.tight_layout()
        plt.show()
    elif n == 4:
        plt.plot(x, y, '-', label="van driest")
        plt.plot(x_exp, y_exp, '*', color="green", label="Experimental")
        x_exp3 = dataframe3.x_comp * (1.0 /0.15)
        y_exp3 = dataframe3.w_stddev 
        plt.plot(x_exp3, y_exp3, '--', color="blue", label="malha fixa")
        x_exp4 = dataframe4.x_comp * (1.0 /0.15)
        y_exp4 = dataframe4.w_stddev 
        plt.plot(x_exp4, y_exp4, '--', color="purple", label="malha fixa")
        plt.legend(loc='best')
        fig.tight_layout()
        plt.show()
    elif n == 5:
        plt.plot(x, y, '-', label="van driest")
        plt.plot(x_exp, y_exp, '*', color="green", label="Experimental")
        x_exp3 = dataframe3.x_comp * (1.0 /0.15)
        y_exp3 = dataframe3.w_stddev 
        plt.plot(x_exp3, y_exp3, '--', color="blue", label="malha fixa")
        x_exp4 = dataframe4.x_comp * (1.0 /0.15)
        y_exp4 = dataframe4.w_stddev 
        plt.plot(x_exp4, y_exp4, '--', color="purple", label="malha fixa")
        x_exp5 = dataframe5.x_comp * (1.0 /0.15)
        y_exp5 = dataframe5.w_stddev 
        plt.plot(x_exp5, y_exp5, '--', color="yellow", label="malha fixa")
        plt.legend(loc='best')
        fig.tight_layout()
        plt.show()
    else:
        plt.plot(x, y, '-', label="Van Driest Cs = 0.2")
        plt.plot(x_exp, y_exp, '*', color="green", label="Experimental")
        plt.legend(loc='best')
        fig.tight_layout()
        plt.show()

"""
Plot for std u
"""
def plot_std_u(dataframe, exp_u_std, dataframe3, dataframe4, dataframe5, n):
   
    x = dataframe.x_comp * (1.0 /0.15)
    y = dataframe.u_stddev              

    x_exp = exp_u_std.x1_exp
    y_exp = exp_u_std.vel_u_exp               
    
    #x_exp1 = experimental1.x_comp * (1.0 /0.15)  
    #y_exp1 = experimental1.u_stddev               

    dif_mymodel = y - y_exp
    L2_norm_mymodel = np.sqrt((dif_mymodel ** 2).sum()) / len(np.array(y))

    #dif_article = y_exp1 - y_exp
    #L2_norm_article = np.sqrt((dif_article ** 2).sum()) / len(np.array(y))
#
    #print("Norma L2 de u_std Cs=0.1 : {0:.3f}% " .format(L2_norm_mymodel * 100))
    #print("Norma L2 de u_std Cs=0.2 : {0:.3f}% " .format(L2_norm_article * 100))
    #print("Norma Loo de u_std Cs=0.1 : {0:.3f}% " .format(np.amax(abs(dif_mymodel.sum()))))
    #print("Norma Loo de u_std Cs=0.2 : {0:.3f}% " .format(np.amax(abs(dif_article.sum()))))

    plt.style.use('ggplot')

    fig = plt.figure(figsize=(6.0, 4.0))
    axes1 = fig.add_subplot(1, 1, 1)
    axes1.set_ylabel('std_u')
    axes1.set_xlabel('x / L')
   
    if n == 3:
        plt.plot(x, y, '-', label="van driest")
        plt.plot(x_exp, y_exp, '*', color="green", label="Experimental")
        x_exp3 = dataframe3.x_comp * (1.0 /0.15)
        y_exp3 = dataframe3.u_stddev 
        plt.plot(x_exp3, y_exp3, '--', color="blue", label="malha fixa")
        plt.legend(loc='best')
        fig.tight_layout()
        plt.show()
    elif n == 4:
        plt.plot(x, y, '-', label="van driest")
        plt.plot(x_exp, y_exp, '*', color="green", label="Experimental")
        x_exp3 = dataframe3.x_comp * (1.0 /0.15)
        y_exp3 = dataframe3.u_stddev 
        plt.plot(x_exp3, y_exp3, '--', color="blue", label="malha fixa")
        x_exp4 = dataframe4.x_comp * (1.0 /0.15)
        y_exp4 = dataframe4.u_stddev 
        plt.plot(x_exp4, y_exp4, '--', color="purple", label="malha fixa")
        plt.legend(loc='best')
        fig.tight_layout()
        plt.show()
    elif n == 5:
        plt.plot(x, y, '-', label="van driest")
        plt.plot(x_exp, y_exp, '*', color="green", label="Experimental")
        x_exp3 = dataframe3.x_comp * (1.0 /0.15)
        y_exp3 = dataframe3.u_stddev 
        plt.plot(x_exp3, y_exp3, '--', color="blue", label="malha fixa")
        x_exp4 = dataframe4.x_comp * (1.0 /0.15)
        y_exp4 = dataframe4.u_stddev 
        plt.plot(x_exp4, y_exp4, '--', color="purple", label="malha fixa")
        x_exp5 = dataframe5.x_comp * (1.0 /0.15)
        y_exp5 = dataframe5.u_stddev 
        plt.plot(x_exp5, y_exp5, '--', color="yellow", label="malha fixa")
        plt.legend(loc='best')
        fig.tight_layout()
        plt.show()
    else:
        plt.plot(x, y, '-', label="Van Driest Cs = 0.2")
        plt.plot(x_exp, y_exp, '*', color="green", label="Experimental")
        plt.legend(loc='best')
        fig.tight_layout()
        plt.show()

"""
Plot for std ke
"""
def plot_std_ke(dataframe, exp_ke_std, dataframe3, dataframe4, dataframe5, n):
    
    x = dataframe.x_comp * (1.0 /0.15)
    y = dataframe.ke_stddev           

    x_exp = exp_ke_std.x1_exp
    y_exp = exp_ke_std.std_ke_exp             
    
    #x_exp1 = experimental1.x_comp * (1.0 /0.15) 
    #y_exp1 = experimental1.ke_stddev             

    dif_mymodel = y - y_exp
    L2_norm_mymodel = np.sqrt((dif_mymodel ** 2).sum()) / len(np.array(y))

    #dif_article = y_exp1 - y_exp
    #L2_norm_article = np.sqrt((dif_article ** 2).sum()) / len(np.array(y))
#
    #print("Norma L2 de ke_std Cs=0.1 : {0:.3f}% " .format(L2_norm_mymodel * 100))
    #print("Norma L2 de ke_std Cs=0.2 : {0:.3f}% " .format(L2_norm_article * 100))
    #print("Norma Loo de ke_std Cs=0.1 : {0:.3f}% " .format(np.amax(abs(dif_mymodel.sum()))))
    #print("Norma Loo de ke_std Cs=0.2 : {0:.3f}% " .format(np.amax(abs(dif_article.sum()))))

    plt.style.use('ggplot')

    fig = plt.figure(figsize=(6.0, 4.0))
    axes1 = fig.add_subplot(1, 1, 1)
    axes1.set_ylabel('std_ke')
    axes1.set_xlabel('x / L')
 
    
    if n == 3:
        plt.plot(x, y, '-', label="van driest")
        plt.plot(x_exp, y_exp, '*', color="green", label="Experimental")
        x_exp3 = dataframe3.x_comp * (1.0 /0.15)
        y_exp3 = dataframe3.ke_stddev 
        plt.plot(x_exp3, y_exp3, '--', color="blue", label="malha fixa")
        plt.legend(loc='best')
        fig.tight_layout()
        plt.show()
    elif n == 4:
        plt.plot(x, y, '-', label="van driest")
        plt.plot(x_exp, y_exp, '*', color="green", label="Experimental")
        x_exp3 = dataframe3.x_comp * (1.0 /0.15)
        y_exp3 = dataframe3.ke_stddev 
        plt.plot(x_exp3, y_exp3, '--', color="blue", label="malha fixa")
        x_exp4 = dataframe4.x_comp * (1.0 /0.15)
        y_exp4 = dataframe4.ke_stddev 
        plt.plot(x_exp4, y_exp4, '--', color="purple", label="malha fixa")
        plt.legend(loc='best')
        fig.tight_layout()
        plt.show()
    elif n == 5:
        plt.plot(x, y, '-', label="van driest")
        plt.plot(x_exp, y_exp, '*', color="green", label="Experimental")
        x_exp3 = dataframe3.x_comp * (1.0 /0.15)
        y_exp3 = dataframe3.ke_stddev 
        plt.plot(x_exp3, y_exp3, '--', color="blue", label="malha fixa")
        x_exp4 = dataframe4.x_comp * (1.0 /0.15)
        y_exp4 = dataframe4.ke_stddev 
        plt.plot(x_exp4, y_exp4, '--', color="purple", label="malha fixa")
        x_exp5 = dataframe5.x_comp * (1.0 /0.15)
        y_exp5 = dataframe5.ke_stddev 
        plt.plot(x_exp5, y_exp5, '--', color="yellow", label="malha fixa")
        plt.legend(loc='best')
        fig.tight_layout()
        plt.show()
    else:
        plt.plot(x, y, '-', label="Van Driest Cs = 0.2")
        plt.plot(x_exp, y_exp, '*', color="green", label="Experimental")
        plt.legend(loc='best')
        fig.tight_layout()
        plt.show()


"""
Plot for std ln_ke
"""
def plot_ln_ke(dataframe, exp_ke_std, dataframe3, dataframe4, dataframe5, n):
    
    x = dataframe.x_comp * (1.0 /0.15)
    y = dataframe.ke_stddev           

    k = np.pi / x
    k1 = 2 * k
    ln_k = np.log(k)
    ln_k1 = np.log(k1)
    ln_E = np.log(y)

    x_exp = exp_ke_std.x1_exp
    y_exp = exp_ke_std.std_ke_exp     

    k_exp = np.pi / (x_exp)
    k_exp2 = 2 * k_exp
    ln_k_exp = np.log(k_exp)
    ln_k_exp2 = np.log(k_exp2)
    ln_E_exp = np.log(y_exp)        
    
    x1 = ln_k1[4:10]
    y1 = ln_E[4:10]

    m, b = sm.least_square(x1, y1, 3)


    #dif_article = y_exp1 - y_exp
    #L2_norm_article = np.sqrt((dif_article ** 2).sum()) / len(np.array(y))
#
    #print("Norma L2 de ke_std Cs=0.1 : {0:.3f}% " .format(L2_norm_mymodel * 100))
    #print("Norma L2 de ke_std Cs=0.2 : {0:.3f}% " .format(L2_norm_article * 100))
    #print("Norma Loo de ke_std Cs=0.1 : {0:.3f}% " .format(np.amax(abs(dif_mymodel.sum()))))
    #print("Norma Loo de ke_std Cs=0.2 : {0:.3f}% " .format(np.amax(abs(dif_article.sum()))))

    plt.style.use('ggplot')

    fig = plt.figure(figsize=(6.0, 4.0))
    axes1 = fig.add_subplot(1, 1, 1)
    axes1.set_ylabel('$\ln{E}$')
    axes1.set_xlabel('$\ln{k}$')
    
    if n == 1:
        plt.plot(x_exp, y_exp, '*', color="green", label="Experimental")
        plt.legend(loc='best')
        fig.tight_layout()
        plt.show()

    elif n == 2:
        plt.plot(ln_k_exp, ln_E_exp, '*', color="green", label="Experimental $\ln{k} = \pi/{\Delta}$")
        #plt.plot(ln_k_exp2, ln_E_exp, '*', color="y", label="Experimental $2\pi/{\Delta}$")
        #plt.plot(ln_k, ln_E, 'o', color="r", label="ln_k x ln_E ")
        plt.plot(ln_k1[4:10], ln_E[4:10], 'o', color="r", label="$\ln{k} = 2\pi/{\Delta}$")
        plt.plot(ln_k[4:], ln_E[4:], '-', color="k", label="$\ln{k} = \pi/{\Delta}$")
        y1 = m * x1 + b
        plt.plot(x1, y1, '--', color="y", label="$\ln{k} = 2\pi/{\Delta}$")
        print("O coeficiente angular é = ", m)
        #plt.plot(ln_k1, ln_E, 'o', color="r", label="$\ln{k} = 2\pi/{\Delta}$")
        #plt.plot(ln_k, ln_E, '--', color="k", label="$\ln{k} = \pi/{\Delta}$")
        plt.legend(loc='best')
        fig.tight_layout()
        plt.show()

    
