from cProfile import label
import glob
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import statistical_module as sm
import statistica_probes as sp

################################################################
# Ricardo Tadeu Oliveira Catta Preta
# e-mail: ricardocatta@gmail.com
################################################################

# As colunas dos arquivos 'surf00001*' estão salvas na seguinte ordem: 
# names=['ct,t,dt,xc,yc,zc,u,v,w,P,rho,mu,dpm_u,dpm_v,dpm_w,dpm_ufluct,dpm_vfluct,dpm_wfluct'])

# obs: Este código precisa estar dentro da pasta probe_points, que fica dentro do output
filenames = sorted(glob.glob('surf00001*.dat')) # chamando e ordenando os arquivos em ordem crescente
# Importando os dados do paraview - w_mean
dataframe = pd.read_csv("computational_atual_cut_20-200.csv", keep_default_na=True)

# Importando os dados do paraview - w_mean
dataframe3 = pd.read_csv("densecolumn_t_C015_cut_10-145.csv", keep_default_na=True)

# Importando os dados do paraview - w_mean
dataframe4= pd.read_csv("computational_atual_28_probes_smag02.csv", keep_default_na=True)

# Importando os dados do paraview - w_mean
dataframe5 = pd.read_csv("computational_atual_28_probes_smag02.csv", keep_default_na=True)

# Importando os dados do paraview - w_mean
dataframe6 = pd.read_csv("rho_paraview_position.csv", keep_default_na=True)

# Importando os dados da velocidade média - w_mean
exp_w_mean = pd.read_csv("article_experimental_w_mean.csv", keep_default_na=True)

# Importando os dados ddo desvio padrão da velocidade - stddev_w
exp_w_std = pd.read_csv("article_experimental_w_std.csv", keep_default_na=True)

# Importando os dados ddo desvio padrão da velocidade - stddev_u
exp_u_std = pd.read_csv("article_experimental_u_std.csv", keep_default_na=True)

# Importando os dados ddo desvio padrão da energia cinética - stddev_ke
exp_ke_std = pd.read_csv("article_experimental_ke_std.csv", keep_default_na=True)

uj = 6
vj = 7
wj = 8  
dpm_wj = 11
j = 8                                           # índice da coluna da variável y    
i = 3                                           # índice da coluna da variável x
k = 1                                           # índice da coluna da variável t
dt = 2
cuti = 20000                                     # corte temporal no sinal da velocidade
cutf = 263000
#cut = 1                                     # corte temporal no sinal da velocidade
sonda = 20                                      # sonda escolhida entre 1 e 34
rho10 = 10

u = []                                          # lista para a velocidade u
v = []                                          # lista para a velocidade v
w = []                                          # lista para a velocidade w  
dpm_w = []          
y = []                                          # lista para a posição y
x = []                                          # lista para a posição x
t = []                                          # lista para o tempo
delta_t = []                                    # lista para o passo de tempo
rho = []                                        # lista para a densidade


sp.var_v_experiment(exp_u_std, exp_w_std, exp_ke_std)

for filename in filenames:
    print(filename)

    data = np.loadtxt(fname=filename, skiprows=1)
    x.append(data[:, i])                         # armazenando a coluna i na lista x
    t.append(data[:, k])                         # armazenando a coluna t0 na lista t       
    delta_t.append(data[cuti:cutf, dt])
    u.append(data[cuti:cutf, uj])
    v.append(data[cuti:cutf, vj])
    w.append(data[cuti:cutf, wj])
    dpm_w.append(data[cuti:cutf, dpm_wj])
    rho.append(data[cuti:cutf, rho10])



#x2, y3, x_exp, y_exp = sp.plot_std_vel(x, w, delta_t, rho, dataframe3, exp_w_std, 'w')



sp.plot_mean_vel(x, w, delta_t, 0, 0, exp_w_mean)
sp.plot_std_vel(x, w, delta_t, rho, 0, exp_w_std, 'w')
sp.plot_std_vel(x, u, delta_t, rho, 0, exp_u_std, 'u')
sp.plot_std_ke(x, u, v, w, delta_t, rho, 0, exp_ke_std)
#sp.plot_ln_ke(x, u, v, w, dataframe, exp_ke_std)
#ke = sp.plot_std_ke(x, u, v, w, dataframe3, exp_ke_std)
#sp.fft_ke(t, ke)