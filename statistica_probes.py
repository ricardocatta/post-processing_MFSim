from cProfile import label
import matplotlib.pylab as plt
from matplotlib.font_manager import FontProperties
import numpy as np
from numpy import fft
from scipy.optimize import curve_fit
from scipy.special import expit
import pandas as pd
import statistical_module as sm
import statistical_modules as stat_m




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

    x2 = np.zeros(len(vel))
    y2 = np.zeros(len(vel))


    # preparando uma média de favre
    favre_num = np.zeros(len(vel))                        
    favre_den = np.zeros(len(vel))

    
    for i in range(len(vel)):
        x2[i] = np.mean(x1[i])
        y2[i] = np.mean(y1[i])    # antigo
        favre_num[i] = np.sum(dt[i] * y1[i])
        favre_den[i] = np.sum(dt[i])
        
    y3 = (favre_num) / favre_den
    
    plt.plot(x2, y3, '-', label="Smagorinsky, Cs = 0.15")
    plt.plot(x_exp, y_exp, '*', color="green", label="Experimental")
    plt.legend(loc='best')
    fig.tight_layout()
    plt.show()

    mean_y3 = np.round(np.mean(y3), 3)
    mean_y_exp = np.round(np.mean(y_exp), 3)
    print("\n velocidade média computacional: \n", mean_y3)
    print("\n velocidade média experimental: \n", mean_y_exp)
    print("\n Erro relativo da velocidade média:{0:.3f}% \n" .format(stat_m.relative_error(mean_y_exp, mean_y3)))
    print("\n Erro estatístico da velocidade média experimental: \n", sm.statistical_error(y_exp,6))
    print("\n Erro estatístico da velocidade média computacional: \n", sm.statistical_error(y3,6))
    print("\n Desvio padrão da velocidade média experimental: \n", np.std(y_exp))
    print("\n Desvio padrão da velocidade média computacional: \n", np.std(y3))
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

    Retorna a energia cinética turbulenta e a posição, para o experimental e o computacional.
    """

    std_u = np.zeros(len(u))
    std_v = np.zeros(len(u))
    std_w = np.zeros(len(u))
    ke = np.zeros(len(u))

    for i in range(len(u)):
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

    x2 = np.zeros(len(u))

    dt = np.array(dt)
    rho = np.array(rho)

    for i in range(len(u)):
        x2[i] = np.mean(x1[i])
    
    plt.plot(x2, y1 , '-', label="Smagorinsky, Cs = 0.15")
    plt.plot(x_exp, y_exp, '*', color="green", label="Experimental")
    plt.legend(loc='best')
    fig.tight_layout()
    plt.show()
    return x_exp, y_exp, x2, y1

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
        std_vel = np.zeros(len(vel))

        dt = np.array(dt)
        rho = np.array(rho)

        for i in range(len(vel)):
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

        x2 = np.zeros(len(vel))

        for i in range(len(vel)):
            x2[i] = np.mean(x1[i])

        print("x2 = ", x2)

        plt.plot(x2, y1 , '-', label="Smagorinsky, Cs = 0.15")
        plt.plot(x_exp, y_exp, '*', color="green", label="Experimental")
        plt.legend(loc='best')
        fig.tight_layout()
        plt.show()

        mean_y1 = np.round(np.mean(y1), 3)
        mean_y_exp = np.round(np.mean(y_exp), 3)
        print("\n mean_std_u computacional: \n", mean_y1)
        print("\n mean_std_u experimental: \n", mean_y_exp)
        print("\n Erro relativo de mean_std_u: {0:.3f}% \n" .format(stat_m.relative_error(mean_y_exp, mean_y1)))
        print("\n Erro estatístico std_u experimental: \n", sm.statistical_error(y_exp,6))
        print("\n Erro estatístico std_u computacional: \n", sm.statistical_error(y1,6))
        print("\n std_std_u experimental: \n", np.std(y_exp))
        print("\n std_std_u computacional: \n", np.std(y1))
    
    elif vel_i == 'w':
        std_vel = np.zeros(len(vel))

        dt = np.array(dt)
        rho = np.array(rho)

        for i in range(len(vel)):
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

        x2 = np.zeros(len(vel))

        for i in range(len(vel)):
            x2[i] = np.mean(x1[i])

        plt.plot(x2, y1 , '-', label="Smagorinsky, Cs = 0.15")
        plt.plot(x_exp, y_exp, '*', color="green", label="Experimental")
        plt.legend(loc='best')
        fig.tight_layout()
        plt.show()

        mean_y1 = np.round(np.mean(y1), 3)
        mean_y_exp = np.round(np.mean(y_exp), 3)
        print("\n mean_std_w computacional: \n", mean_y1)
        print("\n mean_std_w experimental: \n", mean_y_exp)
        print("\n Erro relativo de mean_std_w: {0:.3f}% \n" .format(stat_m.relative_error(mean_y_exp, mean_y1)))
        print("\n Erro estatístico std_w experimental: \n", sm.statistical_error(y_exp,6))
        print("\n Erro estatístico std_w computacional: \n", sm.statistical_error(y1,6))
        print("\n std_std_w experimental: \n", np.std(y_exp))
        print("\n std_std_w computacional: \n", np.std(y1))
        return x2, y1, x_exp, y_exp


def reynolds_tensor(u_vel, v_vel, w_vel):
    """
     Calcula o tensor de Reynolds submalha.

    INPUT:

    - u_vel = componente x da velocidade instantânea;
    - v_vel = componente y da velocidade instantânea;
    - w_vel = componente z da velocidade instantânea;

    OUTPUT:

    Retorna o tensor de Reynolds submalha exato e o adimensionalizado.
    """
    u = np.array(u_vel)
    v = np.array(v_vel)
    w = np.array(w_vel)

    u_bar = np.zeros(len(u_vel))
    v_bar = np.zeros(len(u_vel))
    w_bar = np.zeros(len(u_vel))

    uv_bar = np.zeros(len(u_vel))
    vu_bar = np.zeros(len(u_vel))

    uw_bar = np.zeros(len(u_vel))
    wu_bar = np.zeros(len(u_vel))
    
    wv_bar = np.zeros(len(u_vel))
    vw_bar = np.zeros(len(u_vel))
    

    uu_bar = np.zeros(len(u_vel))
    vv_bar = np.zeros(len(u_vel))
    ww_bar = np.zeros(len(u_vel))
    

    for i in range(len(u_vel)):
        u_bar[i] = np.mean(u[i])
        v_bar[i] = np.mean(v[i])
        w_bar[i] = np.mean(w[i])
        uv_bar[i] = np.mean(u[i]*v[i])
        vu_bar[i] = np.mean(v[i]*u[i])
        uw_bar[i] = np.mean(u[i]*w[i])
        wu_bar[i] = np.mean(w[i]*u[i])
        wv_bar[i] = np.mean(w[i]*v[i])
        vw_bar[i] = np.mean(v[i]*w[i])
        uu_bar[i] = np.mean(u[i]*u[i])
        vv_bar[i] = np.mean(v[i]*v[i])
        ww_bar[i] = np.mean(w[i]*w[i])

    Re_uw = uw_bar - u_bar * w_bar
    Re_wu = wu_bar - w_bar * u_bar

    Re_uv = uv_bar - u_bar * v_bar
    Re_vu = vu_bar - v_bar * u_bar
    
    Re_wv = wv_bar - w_bar * v_bar
    Re_vw = vw_bar - v_bar * w_bar

    Re_uu = uu_bar - u_bar * u_bar
    Re_vv = vv_bar - v_bar * v_bar
    Re_ww = ww_bar - w_bar * w_bar

    print("\n mean_Re_uu = \n", np.mean(Re_uu))
    print("\n mean_Re_vv = \n", np.mean(Re_vv))
    print("\n mean_Re_ww = \n", np.mean(Re_ww))
    print("\n mean_Re_wv = \n", np.mean(Re_wv))
    print("\n mean_Re_vw = \n", np.mean(Re_vw))
    print("\n mean_Re_wu = \n", np.mean(Re_wu))
    print("\n mean_Re_uw = \n", np.mean(Re_uw))
    print("\n mean_Re_uv = \n", np.mean(Re_uv))
    print("\n mean_Re_vu = \n", np.mean(Re_vu))
    print("\n Re_wu = \n", Re_wu)
    print("\n Re_uw = \n", Re_uw)
    print("\n Re_uv = \n", Re_uv)
    print("\n Re_vu = \n", Re_vu)

    Re = np.array([[np.mean(Re_uu), np.mean(Re_uv), np.mean(Re_uw)],
    [np.mean(Re_vu), np.mean(Re_vv), np.mean(Re_vw)], [np.mean(Re_wu), 
    np.mean(Re_wv), np.mean(Re_ww)]])

    Re_adm = Re * 1.0 / np.mean(Re_ww)

    return Re, Re_adm

def Fourier_transform_ke(t,xc,u,v,w):
    """
     Calcula a transformada de Fourier da energia cinética turbulenta.

    INPUT:

    - ul = componente x da flutuação da velocidade;
    - vl = componente y da flutuação da velocidade;
    - wl = componente z da flutuação da velocidade;
    - xc = componente x do espaço;
    - t = tempo.

    OUTPUT:

    Retorna os valores da energia cinética turbulenta e a frequência.
    """
    # Calculando as flutações das componentes da velocidade
    ul = u - np.mean(u)
    vl = v - np.mean(v)
    wl = w - np.mean(w)

    Enel = (ul * ul + vl * vl + wl * wl)/2

    yl = Enel
    nl = len(yl)
    kl = np.arange(nl)
    frql = kl
    frql = frql[range(nl//2)]
    Yl1 = np.fft.fft(yl)
    Yl = np.sqrt(Yl1.real ** 2 + Yl1.imag ** 2)
    Yl = Yl[range(nl//2)]
    return Yl, frql

def plot_espectro():
    """
     Plota a densidade espectral da energia cinética turbulenta.
    """

    font = FontProperties()
    font.set_family('serif')

    x1R = 1
    x2R = 10000
    xR = np.arange(x1R,x2R,10)
    yR = np.exp((-5.0/3.0)*np.log(xR))*10000000

    dir = [""]
    probe = ["surf00001_sonda00025.dat"]


    filelist = []
    for j in range(1):
        for i in range(1):
            Path = dir[j] + probe[i]
            filelist.append(Path)

    print("filelist = ", filelist)

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

    kinetic_energy =[]
    frequency = []

    for i in range(len(filelist)):
        ept = Fourier_transform_ke(t[i],xc[i],u[i],v[i],w[i])
        ept_kinetic.append(ept)

    for i in range(len(filelist)):
        kinetic_energy1,  frequency1 = Fourier_transform_ke(t[i],xc[i],u[i],v[i],w[i])
        kinetic_energy.append(kinetic_energy1)
        frequency.append(frequency1)    


    cor =["darkgray","tab:blue","tab:red","black"]
    plt.figure()

    print("Energia cinética = \n", kinetic_energy)
    print("Frequência = \n", frequency)

    for i in range(len(ept_kinetic)):

        plt.xlabel('$\log{f}$ [Hz]', fontsize=19, fontproperties=font)
        plt.ylabel('$\log{E(f)}$', fontsize=19, fontproperties=font)
        plt.loglog(frequency[i],kinetic_energy[i],color='red',linewidth=0.8)
        p_ret  = plt.loglog(xR,yR,'--',color='black',linewidth=1.5, label='$m = -5/3$')

    ax= plt.gca()	
    ax.set_xlim([1,10000])
    ax.set_ylim([0.00001,100000])	
    ax.legend(title='Coeficiente Angular')
    plt.title('Densidade Espectral de Energia Cinética Turbulenta')	
    plt.grid()
    #plt.savefig('Espectro.png', format='png', dpi=350)
    plt.show()