import numpy as np
from gradient_test import optimizador
class optimizador_direct(optimizador):
    def __init__(self, variables, epsilon, f, iter=100):
        super().__init__(variables, epsilon, f, iter)
    def movexploratory(self,basepoint, delta):
        nextpoint = []
        coordanatess = [basepoint]
        newvalue = True
        #Creacion de las coordenadas 
        for i in range(len(basepoint)):
            point = basepoint.copy()
            point2 = basepoint.copy()
            point[i] += delta[i]
            point2[i] -= delta[i]
            coordanatess.append(point)
            coordanatess.append(point2)
        
        #evaluacion de las coordenadas
        for coordenate in coordanatess:
            nextpoint.append(self.funcion(coordenate))
        
        #Busqueda del min 
        minum = np.argmin(nextpoint)
        if (coordanatess[minum] == basepoint).all():
            newvalue = False
        
        return coordanatess[minum], newvalue

    def patternmove(self,currentbestpoint, lastbestpoint):
        basepoint = currentbestpoint + (currentbestpoint - lastbestpoint)
        return basepoint

    def updatedelta(self,delta, alpha):
        new_delta = delta / alpha
        return new_delta
    def hookejeeves(self, d, alpha):
        cont = 0
        x_inicial = np.array(self.variables)
        delta = np.array(d)
        x_anterior = x_inicial
        x_mejor, flag = self.movexploratory(x_inicial, delta, self.funcion)
        print(x_mejor)
        while np.linalg.norm(delta) > self.epsilon:
            if flag:
                x_point = self.patternmove(x_mejor, x_anterior)
                x_mejor_nuevo, flag = self.movexploratory(x_point, delta, self.funcion)
            else:
                delta = self.updatedelta(delta, alpha)
                x_mejor, flag = self.movexploratory(x_mejor, delta )
                x_point = self.patternmove(x_mejor, x_anterior)
                x_mejor_nuevo, flag = self.movexploratory(x_point, delta )
            #Son dos subprocersos
            if self.funcion(x_mejor_nuevo) < self.funcion(x_mejor):
                flag = True
                x_anterior = x_mejor
                x_mejor = x_mejor_nuevo
            else:
                flag = False

            cont += 1
            print(x_mejor_nuevo)
            print(self.funcion(x_mejor_nuevo))
        print("Num de iteraciones {}".format(cont))
        return x_mejor_nuevo
    def delta1(self,N, scale):
        num = np.sqrt(N + 1) + N - 1
        den = N * np.sqrt(2)
        op = num / den
        return op * scale

    def delta2(self,N, scale):
        num = np.sqrt(N + 1) - 1
        den = N * np.sqrt(2)
        op = num / den
        return op * scale

    def create_simplex(self,initial_point, scale=1.0):
        n = len(initial_point)
        simplex = [np.array(initial_point, dtype=float)] 
        d1 = self.delta1(n, scale)
        d2 = self.delta2(n, scale)
        for i in range(n):
            point = np.array(simplex[0], copy=True)  
            for j in range(n):
                if j == i: 
                    point[j] += d1
                else:
                    point[j] += d2
            simplex.append(point)
        
        simplex_final = np.array(simplex)

        return np.round(simplex_final, 4)
    def findpoints(self,points):
        evaluaciones = [self.funcion(p) for p in points]
        worst = np.argmax(evaluaciones)
        best = np.argmin(evaluaciones)
        indices = list(range(len(evaluaciones)))
        indices.remove(worst)
        second_worst = indices[np.argmax([evaluaciones[i] for i in indices])]
        if second_worst == best:
            indices.remove(best)
            second_worst = indices[np.argmax([evaluaciones[i] for i in indices])]
        return best, second_worst, worst
    def xc_calculation(self,x, indexs):
        m = x[indexs]
        centro = []
        for i in range(len(m[0])):
            suma = sum(p[i] for p in m)
            v = suma / len(m)
            centro.append(v)
        return np.array(centro)
    def stopcondition(self,simplex, xc):
        value = 0
        n = len(simplex)
        for i in range(n):
            value += (((self.funcion(simplex[i]) - self.funcion(xc))**2) / (n + 1))
        return np.sqrt(value)

    # Función principal que implementa el algoritmo Nelder-Mead en n dimensiones
    def neldermeadmead(self,gamma, beta):
        cont = 1
        mov = []
        simplex = self.create_simplex(self.variables)
        mov.append(simplex)
        best, secondworst, worst = self.findpoints(simplex, self.funcion)
        indices = [best, secondworst, worst]
        indices.remove(worst)
        centro = self.xc_calculation(simplex, indices)
        x_r = (2 * centro) - simplex[worst]
        x_new = x_r
        if self.funcion(x_r) < self.funcion(simplex[best]): 
            x_new = ((1 + gamma) * centro) - (gamma * simplex[worst])
        elif self.funcion(x_r) >= self.funcion(simplex[worst]):
            x_new = ((1 - beta) * centro) + (beta * simplex[worst])
        elif self.funcion(simplex[secondworst]) < self.funcion(x_r) and self.funcion(x_r) < self.funcion(simplex[worst]):
            x_new = ((1 - beta) * centro) - (beta * simplex[worst])
        simplex[worst] = x_new
        mov.append(np.copy(simplex))
        stop = self.stopcondition(simplex, centro)
        while stop >= self.epsilon:
            stop = 0
            best, secondworst, worst = self.findpoints(simplex)
            indices = [best, secondworst, worst]
            indices.remove(worst)
            centro = self.xc_calculation(simplex, indices)
            x_r = (2 * centro) - simplex[worst]
            x_new = x_r
            if self.funcion(x_r) < self.funcion(simplex[best]):
                x_new = ((1 + gamma) * centro) - (gamma * simplex[worst])
            elif self.funcion(x_r) >= self.funcion(simplex[worst]):
                x_new = ((1 - beta) * centro) + (beta * simplex[worst])
            elif self.funcion(simplex[secondworst]) < self.funcion(x_r) and self.funcion(x_r) < self.funcion(simplex[worst]):
                x_new = ((1 + beta) * centro) - (beta * simplex[worst])
            simplex[worst] = x_new
            stop = self.stopcondition(simplex, centro)
            print(stop)
            mov.append(np.copy(simplex))
            cont+=1
        return simplex[best]
    
    def step_calculation(self,x):
        x_n=np.array(x)
        mu=0
        stddev=1
        random_value = np.random.normal(mu, stddev)
        return x_n + random_value

    def randomwalk(self):
        x=np.array(self.variables)
        xmejor=x
        cont=0
        while(self.iteracion > cont):
            x_nuevo=self.step_calculation(x)
            if self.funcion(x_nuevo)< self.funcion(xmejor):
                xmejor=x_nuevo
            cont+=1
        return xmejor
