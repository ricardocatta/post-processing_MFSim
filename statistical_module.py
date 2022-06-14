import numpy as np
import matplotlib.pylab as plt
import pandas as pd
################################################################
# Ricardo Tadeu Oliveira Catta Preta
# e-mail: ricardocatta@gmail.com
################################################################

def mean(dataframe, casas):
    """
    Calcula a média do vetor/lista

    INPUT:

    - dataframe = A coluna da lista que quero calcular a média;
    - casas = número de casas decimais que quero usar.

    OUTPUT:

    Retorna a média da variável considerada.
    """
    x = dataframe
    x_mean = np.mean(x)
    return np.round(x_mean, casas)
    
def std(dataframe, casas):
    """
    Calcula o desvio padrão do vetor/lista

    INPUT:

    - dataframe = A coluna da lista que quero calcular o desvio padrão;
    - casas = número de casas decimais que quero usar.

    OUTPUT:

    Retorna o desvio padrão da variável considerada.
    """
    x = dataframe
    N = len(x)
    x_mean = np.mean(x)
    x_mean = np.round(x_mean, casas)
    numerador = np.sum((x - x_mean)**2)
    denominador = N - 1
    desvio_p = np.sqrt(numerador/denominador)
    return np.round(desvio_p, casas)

def statistical_error(dataframe, casas):
    """
    Calcula o erro estatístico do vetor/lista

    INPUT:

    - dataframe = A coluna da lista que quero calcular o erro estatístico;
    - casas = número de casas decimais que quero usar.

    OUTPUT:

    Retorna o erro estatístico da variável considerada.
    """
    x = dataframe
    N = len(x)
    desvio_p = std(dataframe, casas)
    statisticalerror = desvio_p / np.sqrt(N)
    return np.round(statisticalerror, casas)

def erro_associado(dataframe, casas, erro_inst):
    """
    Calcula o erro associado do vetor/lista

    INPUT:

    - dataframe = A coluna da lista que quero calcular o erro associado;
    - casas = Número de casas decimais que quero usar;
    - erro_inst = Valor do erro do instrumento analógigo/digital


    OUTPUT:

    Retorna o erro associado da variável considerada.
    """
    erro_estatistico = statistical_error(dataframe, casas)
    erro_instrumental = erro_inst
    erroassociado = np.sqrt((erro_estatistico ** 2) + (erro_instrumental ** 2))
    return np.round(erroassociado, casas)

def propaga_incerteza_3D(derivada1, derivada2, derivada3, erro1, erro2, erro3):
    """
    Calcula a propagação de incerteza associado do vetor/lista

    INPUT:

    - derivada1 = derivada analítica da variável1 que será calculada;
    - derivada2 = derivada analítica da variável2 que será calculada;
    - derivada3 = derivada analítica da variável3 que será calculada;
    - erro1 = É o valor do erro associado da variável1;
    - erro2 = É o valor do erro associado da variável2.
    - erro3 = É o valor do erro associado da variável3.

    OUTPUT:

    Retorna a propagação de incerteza da variável considerada.
    """
    delta_x1 = erro1
    delta_x2 = erro2
    delta_x3 = erro3
    df1 = derivada1
    df2 = derivada2
    df3 = derivada3
    delta_f = np.sqrt((df1 * delta_x1)**2 + (df2 * delta_x2) **2 + (df3 * delta_x3) **2)
    return delta_f

def covariance(x, y, n):
    """
    Calcula a covariância entre as variáveis x e y.

    INPUT:

    - x = o vetor/lista x;
    - y = o vetor/lista y;
    - n = o número de casas decimais.

    OUTPUT:

    Retorna o valor da covariância.
    """
    x_bar = mean(x, n)
    y_bar = mean(y, n)
    xy = x * y
    xy_bar = mean(xy, n)
    cov = x_bar * y_bar - xy_bar
    return np.round(cov, n)



def least_square(x, y, n):
    """
    Calcula os coeficientes angulares e lineares pelo método dos mínimos quadrados

    INPUT:

    - x = o vetor/lista x;
    - y = o vetor/lista y;
    - n = o número de casas decimais.

    OUTPUT:

    Retorna o coeficiente angular m, e o coeficiente linar b, da reta.
    """
    
    x_bar = mean(x, n)
    y_bar = mean(y, n)
    m = covariance(x, y, n) / covariance(x, x, n)
    b = y_bar - m * x_bar
    m = np.round(m, n)
    b = np.round(b, n)
    return m, b

def squared_error(y, y1, n):
    """
    Calcula o erro quadrático.

    INPUT:

    - y = o vetor/lista y;
    - y1 = o vetor/lista y1 que pertence aos pontos da reta;
    - n = o número de casas decimais.

    OUTPUT:

    Retorna a soma do erro quadrático.
    """
    squared_error1 = (y - y1) ** 2
    squared_error1 = np.sum(squared_error1)
    return np.round(squared_error1, n)

def mean_squared_error(y, n):
    """
    Calcula o erro quadrático médio.

    INPUT:

    - y = o vetor/lista y;
    - n = o número de casas decimais.

    OUTPUT:

    Retorna a soma do erro quadrático médio.
    """
    mean_squared_error1 = (y - mean(y, n)) ** 2
    mean_squared_error1 = np.sum(mean_squared_error1) / len(y)
    return np.round(mean_squared_error1, n)

def root_mean_squared_error(y, n):
    """
    Calcula a raiz quadrada do erro quadrático médio.

    INPUT:

    - y = o vetor/lista y;
    - n = o número de casas decimais.

    OUTPUT:

    Retorna a raiz quadrada do erro quadrático médio.
    """
    y = np.sqrt(mean_squared_error(y, n))
    return np.round(y, n)

def erro_regressao(y, y1, n):
    """
    Calcula o erro da regressão.

    INPUT:

    - y = o vetor/lista y;
    - y1 = o vetor/lista y1 que pertence aos pontos da reta;
    - n = o número de casas decimais.

    OUTPUT:

    Retorna o erro da regressão.
    """
    erro = squared_error(y, y1, n) / mean_squared_error(y, n)
    return np.round(erro, n)

def relative_error(y1, y2, n):
    """
    Calcula o erro relativo.

    INPUT:

    - y1 = o vetor/lista y1 computacional;
    - y2 = o vetor/lista y1 experimental;
    - n = o número de casas decimais.

    OUTPUT:

    Retorna o erro relativo em porcentagem.
    """

    er = np.abs(y1 - y2) / y2

    return np.round(er, n)