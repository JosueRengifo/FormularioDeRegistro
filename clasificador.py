# importamos las librerías que necesitamos
from sklearn.tree import DecisionTreeClassifier,plot_tree,export_graphviz # �rbol de decisi�n para clasificaci�n
from sklearn.model_selection import train_test_split
import pandas
import numpy as np
#import graphviz
from io import StringIO ## for Python 3

#para el servicio web
from flask import Flask,jsonify,render_template

#Inicializa el algoritmo de clasificación y lo entrena con datos de prueba
names = ['herramientas para manejo de la informacion', 'integracion de informacion', 'Anos de experiencia en  sistemas de informacion', 'Bases de Datos', 'informacion clinica', 'informacion salud publica', 'terminologias','class']
namesFeatures = ['herramientas para manejo de la informacion', 'integracion de informacion', 'Anos de experiencia en  sistemas de informacion', 'Bases de Datos', 'informacion clinica', 'informacion salud publica', 'terminologias']
namesTarget=['Alto','Medio','Bajo']
dataframe = pandas.read_csv('data.csv', names=names)
array = dataframe.values
Datos = array[:,0:7]
Clases = array[:,7]
aFit,aTest,bFit,bTest=train_test_split(Datos,Clases) # datos de entrenamiento, datos de prueba, clases de entrenamiento, clases de prueba
num_folds = 10
num_instances = len(Datos)
seed = 7
tree = DecisionTreeClassifier(max_depth=10, random_state=7) # parametros opcionales max_depth=2, random_state=42
tree.fit(aFit, bFit) 
DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_split=None, class_weight=None, presort=False) # parametros opcionales max_depth=2, random_state=42
#DPrueba=array[20:21,0:7]
DPrueba=np.array([[3,1,3,1,3,1,3]])
predict=tree.predict(DPrueba)
print (predict[0])

app = Flask(__name__)
@app.route('/api/<params>')
def Clasificator(params):
    rRaw=str.split(params,',')
    feature=np.array([rRaw])
    classification=tree.predict(feature)
    return jsonify(
        feature=params,
        classification=str(classification[0])
        )
@app.route('/')
def home():
    return render_template('index.html')
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True) 
