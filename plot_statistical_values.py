import glob
import numpy as np
import pandas as pd
import statistical_module as sm
import statistica_probes as sp
import statistical_modules as stat_m
from cProfile import label
from matplotlib import pyplot as plt


# Ricardo Tadeu Oliveira Catta Preta
# e-mail: ricardocatta@gmail.com


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

def load_data(filename, column_indices):
    print("Carregando dados da sonda:", filename)
    data = np.loadtxt(fname=filename, skiprows=1)
    return [data[:, i] for i in column_indices]


def main():
    filenames = sorted(glob.glob('surf00001*.dat'))

    exp_w_mean = pd.read_csv("article_experimental_w_mean.csv", keep_default_na=True)
    exp_w_std = pd.read_csv("article_experimental_w_std.csv", keep_default_na=True)
    exp_u_std = pd.read_csv("article_experimental_u_std.csv", keep_default_na=True)
    exp_ke_std = pd.read_csv("article_experimental_ke_std.csv", keep_default_na=True)

    uj, vj, wj, dpm_wj = 6, 7, 8, 11
    j, i, k, dt = 8, 3, 1, 2
    cuti, cutf = 10000, 263000
    sonda = 20
    rho10 = 10

    data = np.array([load_data(filename, [i, k, dt, uj, vj, wj, dpm_wj, rho10]) for filename in filenames])

    x, t, delta_t, u, v, w, dpm_w, rho = map(list, zip(*data))

    sp.plot_mean_vel(x, w, delta_t, 0, exp_w_mean)
    sp.plot_std_vel(x, w, delta_t, 0, exp_w_std, 'w')
    sp.plot_std_vel(x, u, delta_t, 0, exp_u_std, 'u')
    x_exp, ke_exp, x_comp, ke_comp = sp.plot_std_ke(x, u, v, w, delta_t, 0, exp_ke_std)

    reynolds, reynolds_adm = sp.reynolds_tensor(u, v, w)
    print("\n Tensor de Reynolds : \n", reynolds)
    print("\n Tensor de Reynolds adimensional: \n", reynolds_adm)
    mean_ke_comp = np.round(np.mean(ke_comp), 4)
    mean_ke_exp = np.round(np.mean(ke_exp), 4)
    print("\n Média da Energia cinética turbulenta computacional: \n", mean_ke_comp)
    print("\n Média da Energia cinética turbulenta experimental: \n", mean_ke_exp)
    print("\n Erro relativo da Energia cinética turbulenta: {0:.4f}% \n" .format(stat_m.relative_error(mean_ke_exp, mean_ke_comp)))
    print("\n Erro estatístico da Energia cinética turbulenta experimental: \n", sm.statistical_error(ke_exp, 6))
    print("\n Erro estatístico da Energia cinética turbulenta computacional: \n", sm.statistical_error(ke_comp, 6))
    print("\n Desvio padrão da Energia cinética turbulenta experimental: \n", np.std(ke_exp))
    print("\n Desvio padrão da Energia cinética turbulenta computacional: \n", np.std(ke_comp))

main()
    

