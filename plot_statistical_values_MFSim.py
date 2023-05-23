from cProfile import label
import glob
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import statistical_module as sm
import statistica_probes_MFSim as sp
import statistical_modules as stat_m

################################################################
# Ricardo Tadeu Oliveira Catta Preta
# e-mail: ricardocatta@gmail.com
################################################################
"""
Este código irá carregar o módulo statistica_probes.
As possibilidades de pós processamento aqui desenvolvidas são:
- VELOCIDADE MÉDIA: de algum componente do vetor velocidade;
- DESVIO PADRÃO: de algum componente do vetor velocidade;
- ENERGIA CINÉTICA TURBULENTA;
- TENSOR DE REYNOLDS SUBMALHA EXATO E ADIMENSIONAL.

Coloquei especificamente para plotar a velocidade média w, a energia cinética
turbulenta e os desvios padrões de u e w.

Para outras componentes, basta alterar de acordo com a necessidade.
"""
# As colunas dos arquivos 'surf00001*' estão salvas na seguinte ordem: 
# names=['ct,t,dt,xc,yc,zc,u,v,w,P,rho,mu,dpm_u,dpm_v,dpm_w,dpm_ufluct,dpm_vfluct,dpm_wfluct'])

# obs: Este código precisa estar dentro da pasta probe_points, que fica dentro do output
filenames = sorted(glob.glob('surf00001*.dat')) # chamando e ordenando os arquivos em ordem crescente

uj, vj, wj = 6, 7, 8
j, i, k, dt = 8, 3, 1, 2
cuti, cutf = 10000, 263000
sonda = 20

u, v, w, dpm_w, y, x, t, delta_t, rho = [], [], [], [], [], [], [], [], []  # inicializando as listas


#sp.plot_espectro()

# Função para ler os dados de um arquivo e retornar as colunas desejadas
def read_file(filename):
    print("Carregando dados da sonda:", filename)
    data = np.loadtxt(fname=filename, skiprows=1)
    x.append(data[:, i])                         # armazenando a coluna i na lista x
    t.append(data[:, k])                         # armazenando a coluna t0 na lista t       
    delta_t.append(data[cuti:cutf, dt])
    u.append(data[cuti:cutf, uj])
    v.append(data[cuti:cutf, vj])
    w.append(data[cuti:cutf, wj])
    #dpm_w.append(data[cuti:cutf, dpm_wj])
    #rho.append(data[cuti:cutf, rho10])

# Chama a função read_file para cada arquivo utilizando a função map
list(map(read_file, filenames))

"""
Irá plotar respectivamente a velocidade média w, desvio padrão de w, 
desvio padrão de u e energia cinética turbulenta. Os plotes irão comparar
os resultados simulados com os experimentais.
"""

sp.plot_mean_vel(x, w, delta_t, 0)
sp.plot_std_vel(x, w, delta_t, 0, 'w')
sp.plot_std_vel(x, u, delta_t, 0, 'u')
x_comp, ke_comp = sp.plot_std_ke(x, u, v, w, delta_t, 0)

