import numpy as np
from gradient_test import optimizador
class optimizador_direct(optimizador):
    def __init__(self, variables, epsilon, f, iter=100):
        super().__init__(variables, epsilon, f, iter)
       