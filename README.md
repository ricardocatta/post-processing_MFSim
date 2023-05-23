# post_processing_MFSim

O objetivo do projeto "post_processing_MFSim" é realizar comparações entre os resultados do MFSim e resultados experimentais. 

- DESCRIÇÃO DO PROBLEMA: As sondas utilizadas para o problema computacional é descrita a seguir. O problema consiste de 34 sondas distribuídas ao longo do domínio. Foram fixadas 
as posições y = 0,075 m e z = 0,25 m. As sondas variam somente na posição x, com  0 < x < 0,15 m. Portanto, para a sonda 1, temos um valor para a posição x, com y e z fixos com os 
valores citados anteriormente. Para a sonda 2, temos outro valor de x, mas mantendo as posições y e z; e assim sucessivamente. Para cada uma das 34 sondas, temos valores para as 
velocidades que terão a mesma quantidade de iterações feitas na simulação.

- COMPARAÇÃO COM OS RESULTADOS EXPERIMENTAIS:

-- plot_statistical_values.py: Este é o arquivo principal que você deve executar. Ele chamará o módulo statistica_probes.py, onde estão implementadas as funções relacionadas ao processamento estatístico dos resultados.

-- statistica_probes.py: Este módulo contém as funções necessárias para realizar o processamento estatístico dos resultados do MFSim e dos resultados experimentais. Aqui você encontrará cálculos como médias, desvios padrões e métodos estatísticos.

Caso precise realizar outros cálculos, como cálculo de erro relativo, erro associado, método dos mínimos quadrados e propagação de incertezas, você pode utilizar o módulo:

-- statistical_module.py: Esse módulo contém implementações de funções para cálculos mais avançados, como os mencionados acima.

Portanto, para realizar a comparação dos resultados do MFSim com resultados experimentais, utilize os arquivos mencionados acima e, se necessário, utilize também o módulo statistical_module.py para realizar outros cálculos estatísticos.

- PLOT DOS RESULTADOS, SEM COMPARAÇÃO COM RESULTADOS:

-- plot_statistical_values_MFSim.py

-- statistica_probes_MFSim.py