from cProfile import label
import matplotlib.pylab as plt
from matplotlib.font_manager import FontProperties
import numpy as np
from numpy import fft
from scipy.optimize import curve_fit
from scipy.special import expit
import pandas as pd
import statistical_module as sm



################################################################
# Ricardo Tadeu Oliveira Catta Preta
# e-mail: ricardocatta@gmail.com
################################################################



def plot_mean_vel(x, vel, dt, rho, exp_w_mean):
    """
     Plota o valor médio da velocidade em função da posição.

    INPUT:

    - x = vetor de posição com a quantidade tempo por número de probes;
    - vel = vetor de velocidade com a quantidade tempo por número de probes;
    - dt = vetor com o passo de tempo;
    - rho = vetor com a densidade do campo euleriano;
    - exp_w_mean = tabela com os dados experimentais.

    OUTPUT:

    Retorna o gráfico da velocidade média em função da posição e os componentes
    da posição e velocidade média.
    """
    x_exp = exp_w_mean.x1_exp
    y_exp = exp_w_mean.mean_w_exp

    plt.style.use('ggplot')
        
    fig = plt.figure(dpi=350)                       # resolução da imagem por ponto
    axes1 = fig.add_subplot(1, 1, 1)
    axes1.set_ylabel('mean_w')
    axes1.set_xlabel('x / L')
        
    x0 = np.array(x)                                # adimensionalizando a coordenada x
    x1 = x0 * 1.0 / 0.15                            # para o caso que analisei, admensionalisei um comprimento de 0,15 m entre 0 e 1.
    
    dt = np.array(dt)
    rho = np.array(rho)
    y1 = np.array(vel)

    x2 = np.zeros(34)
    y2 = np.zeros(34)

    # preparando uma média de favre
    favre_num = np.zeros(34)                        
    favre_den = np.zeros(34)

    
    for i in range(34):
        x2[i] = np.mean(x1[i])
        y2[i] = np.mean(y1[i])    # antigo
        favre_num[i] = np.sum(dt[i] * y1[i])
        favre_den[i] = np.sum(dt[i])
        
    y3 = (favre_num) / favre_den
    print("y3 é = ", y3)
    print("favre_num é = ", favre_num)
    print("favre_den é = ", favre_den)
    
    plt.plot(x2, y3, '-', label="Smagorinsky, Cs = 0.15")
    plt.plot(x_exp, y_exp, '*', color="green", label="Experimental")
    plt.legend(loc='best')
    fig.tight_layout()
    plt.show()
    return x2, y3, x_exp, y_exp

def plot_std_ke(x, u, v, w, dt, rho, exp_ke_std):
    """
    Plota o desvio padrão da energia cinética em função da posição.

    INPUT:

    - x = vetor de posição com a quantidade tempo por número de probes;
    - u = vetor de velocidade u com a dimensão da quantidade tempo por número de probes;
    - v = vetor de velocidade v com a dimensão da quantidade tempo por número de probes;
    - w = vetor de velocidade w com a dimensão da quantidade tempo por número de probes;
    - dt = vetor com o passo de tempo;
    - rho = vetor com a densidade do campo euleriano;
    - exp_ke_std = tabela com os dados experimentais para energia cinética turbulenta.

    OUTPUT:

    Retorna o desvio padrão da energia cinética turbulenta em função da posição.
    """

    std_u = np.zeros(34)
    std_v = np.zeros(34)
    std_w = np.zeros(34)
    ke = np.zeros(34)

    for i in range(34):
        std_u[i] = sm.std(u[i], 3)
        std_v[i] = sm.std(v[i], 3)
        std_w[i] = sm.std(w[i], 3)

    ke = (0.5) * np.abs(std_u ** 2 + std_v ** 2 + std_w ** 2)
    
    x_exp = exp_ke_std.x1_exp
    y_exp = exp_ke_std.std_ke_exp

    plt.style.use('ggplot')
        
    fig = plt.figure(dpi=350)
    axes1 = fig.add_subplot(1, 1, 1)
    axes1.set_ylabel('ke')
    axes1.set_xlabel('x / L')
        
    x0 = np.array(x)                                # adimensionalizando a coordenada x
    x1 = x0 * 1.0 / 0.15                            # para o caso que analisei, admensionalisei um comprimento de 0,15 m entre 0 e 1.

    y1 = np.array(ke)

    x2 = np.zeros(34)

    dt = np.array(dt)
    rho = np.array(rho)

    for i in range(34):
        x2[i] = np.mean(x1[i])
    
    plt.plot(x2, y1 , '-', label="Smagorinsky, Cs = 0.15")
    plt.plot(x_exp, y_exp, '*', color="green", label="Experimental")
    plt.legend(loc='best')
    fig.tight_layout()
    plt.show()
    return x2, y1, x_exp, y_exp

def plot_std_vel(x, vel, dt, rho, exp_std_vel, vel_i):
    """
    Plota o valor desvio padrão da velocidade em função da posição.

    INPUT:

    - x = vetor de posição com a quantidade tempo por número de probes;
    - vel = vetor de velocidade com a quantidade tempo por número de probes;
    - dt = vetor com o passo de tempo;
    - rho = vetor com a densidade do campo euleriano;
    - exp_std_vel = tabela com os dados experimentais para o desvio padrão da velocidade;
    - vel_i = tabela de velocidade que se deseja calcular o desvio padrão.

    OUTPUT:

    Retorna um vetor com 34 pontos da velocidade média em função da posição.
    """
    if vel_i == 'u':
        std_vel = np.zeros(34)

        dt = np.array(dt)
        rho = np.array(rho)

        for i in range(34):
            std_vel[i] = sm.std(vel[i], 3)
    
        mean_var_vel = np.abs(std_vel)
    
        x_exp = exp_std_vel.x1_exp
        y_exp = exp_std_vel.std_u_exp

        plt.style.use('ggplot')

        fig = plt.figure(dpi=350)
        axes1 = fig.add_subplot(1, 1, 1)
        axes1.set_ylabel('std_u')
        axes1.set_xlabel('x / L')

        x0 = np.array(x)                                # adimensionalizando a coordenada x
        x1 = x0 * 1.0 / 0.15                            # para o caso que analisei, admensionalisei um comprimento de 0,15 m entre 0 e 1.

        y1 = np.array(mean_var_vel)

        x2 = np.zeros(34)

        for i in range(34):
            x2[i] = np.mean(x1[i])

        print("x2 = ", x2)

        plt.plot(x2, y1 , '-', label="Smagorinsky, Cs = 0.15")
        plt.plot(x_exp, y_exp, '*', color="green", label="Experimental")
        plt.legend(loc='best')
        fig.tight_layout()
        plt.show()
    
    elif vel_i == 'w':
        std_vel = np.zeros(34)

        dt = np.array(dt)
        rho = np.array(rho)

        for i in range(34):
            std_vel[i] = sm.std(vel[i], 3)
    
        mean_var_vel = np.abs(std_vel)
    
        x_exp = exp_std_vel.x1_exp
        y_exp = exp_std_vel.std_w_exp

        plt.style.use('ggplot')

        fig = plt.figure(dpi=350)
        axes1 = fig.add_subplot(1, 1, 1)
        axes1.set_ylabel('std_w')
        axes1.set_xlabel('x / L')

        x0 = np.array(x)                                # adimensionalizando a coordenada x
        x1 = x0 * 1.0 / 0.15                            # para o caso que analisei, admensionalisei um comprimento de 0,15 m entre 0 e 1.

        y1 = np.array(mean_var_vel)

        x2 = np.zeros(34)

        for i in range(34):
            x2[i] = np.mean(x1[i])

        plt.plot(x2, y1 , '-', label="Smagorinsky, Cs = 0.15")
        plt.plot(x_exp, y_exp, '*', color="green", label="Experimental")
        plt.legend(loc='best')
        fig.tight_layout()
        plt.show()
        return x2, y1, x_exp, y_exp