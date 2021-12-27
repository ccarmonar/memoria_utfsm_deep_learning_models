## Implementación de la arquitectura de Neo adaptada para SPARQL

### Usage
El jupyter notebook ``ModelTreeConvSparql.ipynb`` permite entrenar el modelo propuesto y evaluarlo para el test set.
Este primeramente divide los datos de entrenamiento, aplica algunos métodos de limpieza y preparación de los datos.
Luego crea instancia un objeto de la clase ``NeoRegression`` para el entrenamiento. Finalmente, evalúa con este mismo 
objeto para el test set y grafica los resultados.

La clase ``NeoRegression`` presente en [model_trees_algebra.py](model_trees_algebra.py), contiene las funciones para el
 preparación de los datos, entrenamiento y evaluación del modelo.
 
La implementación de la arquitectura del modelo puede ser vista en [NeoNet](net.py). Está compuesta por la capas 
a nivel de consulta y las capas convoluciones sobre árboles.

En la carpeta TreeConvolution se encuentra la implementación de las capas para el TCNN. Este código original se definió
en el proyecto [@github.com:learnedsystems/BaoForPostgreSQL/TreeConvolution](https://github.com/learnedsystems/BaoForPostgreSQL/tree/master/bao_server/TreeConvolution).
En tcnn.py se puede encontrar ahí la implementación de:
 
 - ``BinaryTreeConv``: Implementación de la capa convolutional del TCNN utilizando una``nn.Conv1D``.
 - ``TreeActivation``: Función de activación.
 - ``TreeLayerNorm``: Implementación de la capa de normalización  aplicada luego de ``BinaryTreeConv``.
 - ``DynamicPooling``: Puling dinámico para convertir valores de la última capa convolucional en una vector de tamaño fijo.
 - ``BinaryTreeConvWithQData``: Nuestra implementación de la concatenación de las características a nivel de consulta
  con las características a nivel de plan. Esta se utiliza com la primera capa de convolución sobre árboles.
  Ver [BinaryTreeConvWithQData.py](TreeConvolution/tcnn.py)
 
### Requirements.
Probado en un server con una GPU (Nvidia 2080ti). 
La red está implementada en ``pytorch`` y utiliza otras bibliotecas como: ``pandas``,``numpy``,``plotly``,``matplotlib``, ``sklearn``.
