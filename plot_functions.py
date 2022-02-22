import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import statistical_modules as sms
import statistical_module as sm

# Importando os dados do paraview - w_mean
dataframe = pd.read_csv("computational_atual_cut_20-200.csv", keep_default_na=True)

# Importando os dados do paraview - w_mean
dataframe3 = pd.read_csv("densecolumn_t_C015_cut_10-145.csv", keep_default_na=True)

# Importando os dados do paraview - w_mean
dataframe4= pd.read_csv("computational_atual_28_probes_smag02.csv", keep_default_na=True)

# Importando os dados do paraview - w_mean
dataframe5 = pd.read_csv("computational_atual_28_probes_smag02.csv", keep_default_na=True)

# Importando os dados da velocidade média - w_mean
exp_w_mean = pd.read_csv("article_experimental_w_mean.csv", keep_default_na=True)

# Importando os dados ddo desvio padrão da velocidade - stddev_w
exp_w_std = pd.read_csv("article_experimental_w_std.csv", keep_default_na=True)

# Importando os dados ddo desvio padrão da velocidade - stddev_u
exp_u_std = pd.read_csv("experimental_x_computational_std_u2.csv", keep_default_na=True)

# Importando os dados ddo desvio padrão da energia cinética - stddev_ke
exp_ke_std = pd.read_csv("article_experimental_ke_std.csv", keep_default_na=True)

# número de gráficos
n = 2

#sms.plot_mean_w(dataframe3, exp_w_mean, dataframe3, dataframe4, dataframe5, n) 
#sms.plot_std_w( dataframe3 , exp_w_std, dataframe3, dataframe4, dataframe5, n)
#sms.plot_std_u( dataframe3 , exp_u_std, dataframe3, dataframe4, dataframe5, n)
#sms.plot_std_ke(dataframe3, exp_ke_std, dataframe3, dataframe4, dataframe5, n)
sms.plot_ln_ke(dataframe, exp_ke_std, dataframe3, dataframe4, dataframe5, n)
