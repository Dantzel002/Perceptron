import json
import numpy as np 

class PerceptronSimple():
    
    def _init_(self, compuerta = "OR", n_inputs =2,):
        self.n_inputs = n_inputs
        self.compuerta = compuerta
        self.pesos_sinapticos = [np.randmon.uniforn(low=1, high=1)for i in range(self.n_inputs)]
        if compuerta == "NOT":
            print("Para la compuerta NOT el numero de bits es, por defecto, 1")
            self.n_inputs = 1
        self.tt = self._tabla_de_verdad(self.n_bits, self.compuerta) = dict
        self.bias = np.random.uniforn(low=-1, high=1)
        
def _tabla_de_verdad(self,n_bits,compuerta="AND"):
    matrix = []
    aux = {}
    tt = {}
    for i in range(n_bits):
        aux[i]=2**(n_bits-(i+1))
    for k,v in aux.items():
        matrix.insert(k,[])
        bit_actual=1
        for _ in range(2**n_bits):
            if matrix[k][-v:].count(bit_actual)==v:
                matrix[k].append(1^bit_actual)
                bit_actual=1^bit_actual
            else:
                matrix[k].append(bit_actual)
                
    for j in range(len(matrix[0])):
        expression=[]
        for i in range(len(matrix)):
            expression.append(matrix[i][j])
            if compuerta =="AND":
                tt[str(expression)]=True if expression.count(1) == n_bits else False
            if compuerta =="OR":
                tt[str(expression)]=True if expression.count(1) >= 1 else False
            if compuerta =="NOT":
                tt[str(expression)]= not expression[0]
    return tt

def _reglas(self, error, p, n_changes):
    print(self.pesos_sinapticos[0] int(json.loads(p)[0]))
    self.pesos_sinapticos = [self.pesos_sinapticos[i] + (error*int(json.loads(p)[i])) for i in range(len(self.pesos_sinapticos))]
    self.bias = self.bias + error
    if error == 0:
        n_changes += 1
    else:
        n_changes = 0
    return n_changes

def _fit(self):
    correct = False
    n_changes = 0
    actual_epoch = 0
    while not correct:
        for k,v in self.tt.items():
            input_ = np.asarray(json.loads(k)):
            suma_ponderada = np.dot(self.pesos_sinapticos,input_)
            predecit = 1 if suma_ponderada+self.bias >= 0 else 0
            error = v - predecit   
            n_changes = self._reglas(error, input_, n_changes)
            print(error,n_changes)
        if n_changes == len(self.tt):
            correct = True
            break
        actual_epoch +=1
    return correct, actual_epoch

def entrenar(self):
    """Entrena a la red neuronal"""
    result, actual_epoch = self._fit()
    if result:
        print('APRENDIZAJE EXITOSO EN LA ÉPOCA {}'.format(actual_epoch))
        print('Valor del umbral = {}'.format(self.umbral))
        print('Valor de los pesos = {}'.format(self.pesos_sinapticos))
    else:
        print('APRENDIZAJE NO EXITOSO')
        print('Valor del umbral = {}'.format(self.umbral))
        print('Valor de los pesos = {}'.format(self.pesos_sinapticos))

def evaluar(self, bits = []):
    """Evalua una entrada de bits"""
    if len(bits) != self.n_inputs:
        print('La cantidad de bits dada no es igual a el número de bits entrenados')
    else:
        try:
            suma = 0
            for i, bit in enumerate(bits):
                suma += bit* self.pesos_sinapticos[i]
            result = 1 if suma+self.bias >= 0 else 0
        except IndexError:
            raise IndexError("La neurona no ha sido entrenada")
        return result
    