import matplotlib.pylab as plt
from matplotlib.font_manager import FontProperties
import numpy as np
from numpy import fft
import pandas as pd
import statistical_module as sm



################################################################
# Ricardo Tadeu Oliveira Catta Preta
# e-mail: ricardocatta@gmail.com
################################################################



def plot_mean_vel(x, vel, dt, rho, dataframe, exp_w_mean):
    """
     Plota o valor médio da velocidade em função da posição.

    INPUT:

    - x = vetor de posição com a quantidade tempo por número de probes;
    - vel = vetor de velocidade com a quantidade tempo por número de probes;
    - dt = vetor com o passo de tempo;
    - rho = vetor com a densidade do campo euleriano;
    - dataframe = tabela dos dados coletados pelo paraview;
    - exp_w_mean = tabela com os dados experimentais.

    OUTPUT:

    Retorna o gráfico da velocidade média em função da posição.
    """
    x_paraview = dataframe.x_comp * (1.0 /0.15)
    y_paraview = dataframe.w_average

    rho_paraview = dataframe.density_average
    w_paraview = []
    for i in range(29):
        w_paraview.append(np.sum((y_paraview[i] * rho_paraview[i])) / np.sum(rho_paraview[i])) # média de ponderada pelo dt.

    x_exp = exp_w_mean.x1_exp
    y_exp = exp_w_mean.mean_w_exp
    #y_exp = exp_w_mean.dpm_mean_w


    plt.style.use('ggplot')
        
    fig = plt.figure(figsize=(10.0, 4.0))
    axes1 = fig.add_subplot(1, 1, 1)
    axes1.set_ylabel('mean_dpm_w')
    axes1.set_xlabel('x / L')
        
    x0 = np.array(x)                                # adimensionalizando a coordenada x
    x1 = x0 * 1.0 / 0.15
    
    dt = np.array(dt)
    rho = np.array(rho)
    y1 = np.array(vel)

    x2 = np.zeros(34)
    y2 = np.zeros(34)

    favre_num = np.zeros(34)
    favre_den = np.zeros(34)

    
    for i in range(34):
        x2[i] = np.mean(x1[i])
        y2[i] = np.mean(y1[i])    # antigo
        #favre_num[i] = np.sum((dt[i] * y1[i] * rho[i]))
        #favre_den[i] = np.sum(dt[i] * rho[i])
        favre_num[i] = np.sum(dt[i] * y1[i])
        favre_den[i] = np.sum(dt[i])
        
    y3 = (favre_num) / favre_den
    print("y3 é = ", y3)
    print("favre_num é = ", favre_num)
    print("favre_den é = ", favre_den)
    
    plt.plot(x2, y3, '-', label="Smag Cs = 0.15")
    #plt.plot(x2, y3 , 'o', label="Smag Cs = 0.15 (com favre)")
    #plt.plot(x_paraview, w_paraview, '--', label="Smag Cs = 0.15 (paraview)")
    plt.plot(x_exp, y_exp, '*', color="green", label="Experimental")
    plt.legend(loc='best')
    fig.tight_layout()
    plt.show()

def plot_std_ke(x, u, v, w, dt, rho, dataframe, exp_ke_std):
    """
    Plota o desvio padrão da energia cinética em função da posição.

    INPUT:

    - x = vetor de posição com a quantidade tempo por número de probes;
    - u = vetor de velocidade u com a dimensão da quantidade tempo por número de probes;
    - v = vetor de velocidade v com a dimensão da quantidade tempo por número de probes;
    - w = vetor de velocidade w com a dimensão da quantidade tempo por número de probes;
    - dt = vetor com o passo de tempo;
    - rho = vetor com a densidade do campo euleriano;
    - dataframe = tabela dos dados coletados pelo paraview;
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

    x_parav = dataframe.x_comp * (1.0 /0.15)
    y_parav = dataframe.ke_stddev  

    plt.style.use('ggplot')
        
    fig = plt.figure(figsize=(10.0, 4.0))
    axes1 = fig.add_subplot(1, 1, 1)
    axes1.set_ylabel('std_ke')
    axes1.set_xlabel('x / L')
        
    x0 = np.array(x)                                # adimensionalizando a coordenada x
    x1 = x0 * 1.0 / 0.15

    y1 = np.array(ke)

    x2 = np.zeros(34)

    favre_num = np.zeros(34)
    favre_den = np.zeros(34)
    dt = np.array(dt)
    rho = np.array(rho)

    for i in range(34):
        x2[i] = np.mean(x1[i])
        favre_num[i] = np.sum((dt[i] * y1[i] * rho[i]))
        favre_den[i] = np.sum(dt[i] * rho[i])

    y3 = favre_num / favre_den
    print("y3 é = ", y3)

    
    plt.plot(x2, y1 , '-', label="Smag Cs = 0.15")
    #plt.plot(x2, y3 , 'o', label="Smag Cs = 0.15 (com favre)")
    #plt.plot(x_parav, y_parav, '--', label="Smag Cs = 0.15 (paraview)")
    plt.plot(x_exp, y_exp, '*', color="green", label="Experimental")
    plt.legend(loc='best')
    fig.tight_layout()
    plt.show()
    return y1


def plot_std_vel(x, vel, dt, rho, dataframe, exp_std_vel, vel_i):
    """
    Plota o valor desvio padrão da velocidade em função da posição.

    INPUT:

    - x = vetor de posição com a quantidade tempo por número de probes;
    - vel = vetor de velocidade com a quantidade tempo por número de probes;
    - dt = vetor com o passo de tempo;
    - rho = vetor com a densidade do campo euleriano;
    - dataframe = tabela dos dados coletados pelo paraview;
    - exp_std_vel = tabela com os dados experimentais para o desvio padrão da velocidade;
    - vel_i = tabela de velocidade que se deseja calcular o desvio padrão.

    OUTPUT:

    Retorna um vetor com 34 pontos da velocidade média em função da posição.
    """
    if vel_i == 'u':
        std_vel = np.zeros(34)

        favre_num = np.zeros(34)
        favre_den = np.zeros(34)
        dt = np.array(dt)
        rho = np.array(rho)

        for i in range(34):
            std_vel[i] = sm.std(vel[i], 3)
        
        for i in range(34):
            favre_num[i] = np.sum((dt[i] * std_vel[i] * rho[i]))
            favre_den[i] = np.sum(dt[i] * rho[i])

        y3 = favre_num / favre_den
    
        mean_var_vel = np.abs(std_vel)
    

        x_exp = exp_std_vel.x1_exp
        y_exp = exp_std_vel.std_u_exp

        x_paraview = dataframe.x_comp * (1.0 /0.15)
        y_paraview = dataframe.u_stddev  

        plt.style.use('ggplot')

        fig = plt.figure(figsize=(10.0, 4.0))
        axes1 = fig.add_subplot(1, 1, 1)
        axes1.set_ylabel('std_u')
        axes1.set_xlabel('x / L')

        x0 = np.array(x)                                # adimensionalizando a coordenada x
        x1 = x0 * 1.0 / 0.15

        y1 = np.array(mean_var_vel)

        x2 = np.zeros(34)

        for i in range(34):
            x2[i] = np.mean(x1[i])

        print("x2 = ", x2)

        plt.plot(x2, y1 , '-', label="Smag Cs = 0.15")
        #plt.plot(x2, y3 , 'o', label="Smag Cs = 0.15 (com favre)")
        #plt.plot(x_paraview, y_paraview, '--', label="Smag Cs = 0.15 (paraview)")
        plt.plot(x_exp, y_exp, '*', color="green", label="Experimental")
        plt.legend(loc='best')
        fig.tight_layout()
        plt.show()
    
    elif vel_i == 'w':
        std_vel = np.zeros(34)

        favre_num = np.zeros(34)
        favre_den = np.zeros(34)
        dt = np.array(dt)
        rho = np.array(rho)

        for i in range(34):
            std_vel[i] = sm.std(vel[i], 3)
        
        
        for i in range(34):
            favre_num[i] = np.sum((dt[i] * std_vel[i] * rho[i]))
            favre_den[i] = np.sum(dt[i] * rho[i])

        y3 = favre_num / favre_den

    
        mean_var_vel = np.abs(std_vel)
    

        x_exp = exp_std_vel.x1_exp
        y_exp = exp_std_vel.std_w_exp

        x_paraview = dataframe.x_comp * (1.0 /0.15)
        y_paraview = dataframe.w_stddev  

        plt.style.use('ggplot')

        fig = plt.figure(figsize=(10.0, 4.0))
        axes1 = fig.add_subplot(1, 1, 1)
        axes1.set_ylabel('std_w')
        axes1.set_xlabel('x / L')

        x0 = np.array(x)                                # adimensionalizando a coordenada x
        x1 = x0 * 1.0 / 0.15

        y1 = np.array(mean_var_vel)

        x2 = np.zeros(34)

        for i in range(34):
            x2[i] = np.mean(x1[i])

        print("x2 = ", x2)

        plt.plot(x2, y1 , '-', label="Smag Cs = 0.15")
        #plt.plot(x2, y3 , 'o', label="Smag Cs = 0.15 (com favre)")
        #plt.plot(x_paraview, y_paraview, '--', label="Smag Cs = 0.15 (paraview)")
        plt.plot(x_exp, y_exp, '*', color="green", label="Experimental")
        plt.legend(loc='best')
        fig.tight_layout()
        plt.show()

    

def espectro(t,xc,u,v,w):
    """
    Calcula o valor da densidade espectral de energia cinética turbulenta.

    INPUT:

    - t = vetor do tempo;
    - xc = vetor de posição com a quantidade tempo por número de probes;
    - u = vetor de velocidade u com a dimensão da quantidade tempo por número de probes;
    - v = vetor de velocidade v com a dimensão da quantidade tempo por número de probes;
    - w = vetor de velocidade w com a dimensão da quantidade tempo por número de probes.

    OUTPUT:

    - Yl = O vetor com a FFT da energia cinética;
    - frql = O vetor com os valores para a frequência.
    """
    ul = u - np.mean(u)    # Calcula a flutuação da velocidade u
    vl = v - np.mean(v)    # Calcula a flutuação da velocidade v  
    wl = w - np.mean(w)    # Calcula a flutuação da velocidade w  

    Enel = (ul * ul + vl * vl + wl * wl) / 2  

    contUl = 0
    for i in u:
        if contUl<45:
            Enel[contUl] = Enel[contUl]*np.sin(contUl*2*np.pi/180)
        if (contUl>(len(xc)-46)):
            Enel[contUl] = Enel[contUl]*np.sin((2*(45 + contUl - (len(xc)-46)))*np.pi/180)
        contUl = contUl + 1

    print(" contUl = ", contUl)

    yl = Enel
    nl = len(yl)
    kl = np.arange(nl)
    frql = kl
    frql = frql[range(nl//2)]
    Yl1 = np.fft.fft(yl)
    Yl = Yl1.real ** 2 + Yl1.imag ** 2
    Yl = Yl[range(nl//2)]
    print(" Yl = ", Yl)
    print(" frql = ", frql)
    return Yl, frql

def plot_spectral_density():
    """
    Plota a densidade espectral de energia cinética turbulenta.

    OUTPUT:

    Retorna o gráfico da densidade espectral de energia cinética turbulenta.
    """
    font = FontProperties()
    font.set_family('serif')

    x1R = 1
    x2R = 100000
    xR = np.arange(x1R,x2R,10)
    yR = np.exp((-5.0/3.0)*np.log(xR))*1000000000
    yRl = np.exp((-25.0/3.0)*np.log(xR))* (10 ** 34)

    dir = [""]
    probe = ["surf00001_sonda00017.dat"]    # Escolha da probe


    filelist = []
    for j in range(1):
        for i in range(1):
            Path = dir[j] + probe[i]
            filelist.append(Path)

    print(filelist)

    pularLinhas= 10000 #cerca de 10s

    t = []
    xc = []
    u = []
    v = []
    w = []

    # filelist=[Path,Path2,Path3]
    for file in filelist:
        [t0,x,u0,v0,w0] = np.loadtxt(file,unpack=True,skiprows=pularLinhas,usecols=(1,3,6,7,8))
        t.append(t0)
        xc.append(x)
        u.append(u0)
        v.append(v0)
        w.append(w0)


    ept_kinetic = []

    for i in range(len(filelist)):
        ept = espectro(t[i],xc[i],u[i],v[i],w[i])
        ept_kinetic.append(ept)

    plt.figure()
    print(len(ept_kinetic))

    for i in range(len(ept_kinetic)):

        plt.xlabel('$\log{f}$ [Hz]', fontsize=19, fontproperties=font)
        plt.ylabel('$\log{E(f)}$', fontsize=19, fontproperties=font)
        plt.loglog(ept_kinetic[i][1],abs(ept_kinetic[i][0]),color='red',linewidth=0.8)
        plt.loglog(xR,yR,'--',color='black',linewidth=1.5, label='$m = -5/3$')
        plt.loglog(xR,yRl,'--',color='green',linewidth=1.5, label='$m = -25/3$')


    ax= plt.gca()	
    ax.set_xlim([1,100000])
    ax.set_ylim([0.00001,10000000])	
    ax.legend(title='Coeficiente Angular')
    plt.title('Densidade Espectral de Energia Cinética Turbulenta')	
    plt.grid()
    plt.savefig('Espectro_final.png', format='png', dpi=350)
    plt.show()