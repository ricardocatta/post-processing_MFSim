# post_processing_MFSim

O objetivo do projeto "post_processing_MFSim" é realizar comparações entre os resultados do MFSim e resultados experimentais. Para fazer essa comparação, você deve usar os seguintes arquivos:

- plot_statistical_values.py: Este é o arquivo principal que você deve executar. Ele chamará o módulo statistica_probes.py, onde estão implementadas as funções relacionadas ao processamento estatístico dos resultados.

- statistica_probes.py: Este módulo contém as funções necessárias para realizar o processamento estatístico dos resultados do MFSim e dos resultados experimentais. Aqui você encontrará cálculos como médias, desvios padrões e métodos estatísticos.

Caso precise realizar outros cálculos, como cálculo de erro relativo, erro associado, método dos mínimos quadrados e propagação de incertezas, você pode utilizar o módulo:

- statistical_module.py: Esse módulo contém implementações de funções para cálculos mais avançados, como os mencionados acima.

Portanto, para realizar a comparação dos resultados do MFSim com resultados experimentais, utilize os arquivos mencionados acima e, se necessário, utilize também o módulo statistical_module.py para realizar outros cálculos estatísticos.