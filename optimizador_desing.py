import numpy as np

class optimizador:
    def __init__(self,f,epsilon,iter=100):
        self.funcion=f
        self.epsilon=epsilon
        self.iteraciones=iter

class optimizador_univariable(optimizador):
    def __init__(self,x_inicial ,f, epsilon, iter=100):
        super().__init__(f, epsilon, iter)
        self.valor_inicial=x_inicial

class optimizador_multivariable(optimizador):
    def __init__(self,variables ,f, epsilon, iter=100):
        super().__init__(f, epsilon, iter)
        self.variables=variables

class by_regions_elimination(optimizador_univariable):
    def __init__(self, x_inicial, x_limite,f, epsilon, iter=100):
        super().__init__(x_inicial, f, epsilon, iter)
        self.limite=x_limite

class derivative_methods(optimizador_univariable):
    def __init__(self, x_inicial, f, epsilon, iter=100):
        super().__init__(x_inicial, f, epsilon, iter)

class direct_methods(optimizador_multivariable):
    def __init__(self, variables, f, epsilon, iter=100):
        super().__init__(variables, f, epsilon, iter)

class gradient_methods(optimizador_multivariable):
    def __init__(self, variables, f, epsilon, iter=100):
        super().__init__(variables, f, epsilon, iter)

class interval(by_regions_elimination):
    def __init__(self, x_inicial, x_limite, f, epsilon):
        super().__init__(x_inicial, x_limite, f, epsilon)
    
    def findregions(self,rangomin,rangomax,x1,x2):
        if self.funcion(x1)> self.funcion(x2):
            rangomin=rangomin
            rangomax=x2
        elif self.funcion(x1)< self.funcion(x2):
            rangomin=x1
            rangomax=rangomax
        elif self.funcion(x1)== self.funcion(x2):
            rangomin=x1
            rangomax=x2
        return rangomin,rangomax

    def intervalstep3(self,b,x1,xm):
        if self.funcion(x1)< self.funcion(xm):
            b=xm
            xm=x1
            return b,xm,True
        else:
            return b,xm,False

    def intervalstep4(self,a,x2,xm):
        if  self.funcion(x2)<self.funcion (xm):
            a=xm
            xm=x2
            return a,xm,True
        else:
            return a,xm,False

    def intervalstep5(self,b,a):
        l=b-a
        #print(" Valor actual de a y b = {} , {}".format(a,b))
        if abs(l) < self.epsilon : 
            return False
        else:
            return True

    def intervalhalvingmethod(self):
        a,b=self.valor_inicial,self.limite
        xm=(a+b)/2
        l=b-a
        x1=a + (l/4)
        x2=b - (l/4)
        a,b=self.findregions(a,b,x1,x2)
        #Validaciones
        endflag=self.intervalstep5(a,b)
        l=b-a
        while endflag:
            x1=a + (l/4)
            x2=b - l/4
            #Se obtiene las f(x) de x1 y x2 
            b,xm,flag3=self.intervalstep3(b,x1,xm)
            a,xm,flag4=self.intervalstep4(a,x2,xm)
            if flag3== True:
                endflag=self.intervalstep5(a,b)
            elif flag3==False:
                a,xm,flag4=self.intervalstep4(a,x2,xm)
            
            if flag4==True:
                endflag=self.intervalstep5(a,b)
            elif flag4==False: 
                a=x1
                b=x2
                endflag=self.intervalstep5(a,b)
        return xm

class fibonacci(by_regions_elimination):
    def __init__(self, x_inicial, x_limite, f, iter=100):
        super().__init__(x_inicial, x_limite, f, iter)
    def findregions(self,rangomin,rangomax,x1,x2):
        if self.funcion(x1)> self.funcion(x2):
            rangomin=rangomin
            rangomax=x2
        elif self.funcion(x1)< self.funcion(x2):
            rangomin=x1
            rangomax=rangomax
        elif self.funcion(x1)== self.funcion(x2):
            rangomin=x1
            rangomax=x2
        return rangomin,rangomax

    def fibonacci_iterativo(self,n):
        fibonaccie = [0, 1]
        for i in range(2, n):
            fibonaccie.append(fibonaccie[i-1] + fibonaccie[i-2])
        return fibonaccie

    def calculo_lk(self,fibonacci,n,k):
        indice1=n - (k + 1)
        indice2= n + 1
        return fibonacci[indice1]/ fibonacci[indice2]

    def fibonaccisearch(self):
        a,b=self.valor_inicial,self.limite
        n=self.iteraciones
        l=b-a
        seriefibonacci=self.fibonacci_iterativo(n*10)
        #calculo de lk
        k=2
        lk=self.calculo_lk(seriefibonacci,n,k)
        x1=a+lk
        x2=b-lk
        while k != n:
            if k % 2 == 0:
                evalx1=self.funcion(x1)
                a,b=self.findregions(a,b,evalx1,x2)
                #print(" Valor actual de a y b = {} , {}".format(a,b))
            else:
                evalx2=self.funcion(x2)
                a,b=self.findregions(a,b,x1,evalx2)
                #print(" Valor actual de a y b = {} , {}".format(a,b))
            k+=1
        
        return a , b

class goldensearch(by_regions_elimination):
    def __init__(self, x_inicial, x_limite, f, epsilon):
        super().__init__(x_inicial, x_limite, f, epsilon)
    
    def findregions(self,x1, x2, fx1, fx2, a, b):
        if fx1 > fx2:
            return x1, b
        if fx1 < fx2:
            return a, x2
        return x1, x2 
    def w_to_x(self,w, a, b):
        return w * (b - a) + a 

    def busquedaDorada(self):
        a,b=self.valor_inicial,self.limite
        phi = (1 + np.sqrt(5)) / 2 - 1
        aw, bw = 0, 1
        Lw = 1
        k = 1
        while Lw > self.epsilon:
            w2 = aw + phi * Lw
            w1 = bw - phi * Lw
            aw, bw = self.findregions(w1, w2, self.funcion(self.w_to_x(w1, a, b)), self.funcion(self.w_to_x(w2, a, b)), aw, bw)
            k += 1
            Lw = bw - aw
        return (self.w_to_x(aw, a, b) + self.w_to_x(bw, a, b)) / 2

class newton_raphson(derivative_methods):
    def __init__(self, x_inicial, f, epsilon):
        super().__init__(x_inicial, f, epsilon)
    
    def primeraderivadanumerica(self, x_actual):
        delta = 0.0001
        numerador = self.funcion(x_actual + delta) - self.funcion(x_actual - delta) 
        return numerador / (2 * delta)

    def segundaderivadanumerica(self, x_actual):
        delta = 0.0001
        numerado = self.funcion(x_actual + delta) - (2 * self.funcion(x_actual)) + self.funcion(x_actual - delta)
        return numerado / (delta**2)

    def newton_raphson(self):
        k = 1
        x_actual = self.valor_inicial
        #print(f"Iteración {k}: x_actual = {x_actual}")
        
        xderiv = self.primeraderivadanumerica(x_actual)
        xderiv2 = self.segundaderivadanumerica(x_actual)
        xsig = x_actual - (xderiv / xderiv2)
        
        while abs(self.primeraderivadanumerica(xsig)) > self.epsilon:
            x_actual = xsig
            xderiv = self.primeraderivadanumerica(x_actual)
            xderiv2 = self.segundaderivadanumerica(x_actual)
            xsig = x_actual - ((xderiv) /(xderiv2))
        
        return xsig

class biseccion(derivative_methods):
    def __init__(self, x_inicial,limite, f, epsilon):
        super().__init__(x_inicial, f, epsilon)
        self.limite=limite
    def primeraderivadanumerica(self, x_actual):
        delta = 0.0001
        numerador = self.funcion(x_actual + delta) - self.funcion(x_actual - delta) 
        return numerador / (2 * delta)

    def segundaderivadanumerica(self, x_actual):
        delta = 0.0001
        numerado = self.funcion(x_actual + delta) - (2 * self.funcion(x_actual)) + self.funcion(x_actual - delta)
        return numerado / (delta**2)
    
    def biseccionmethod(self):
        a = np.random.uniform(self.valor_inicial, self.limite)
        b = np.random.uniform(self.valor_inicial, self.limite)
        while(self.primeraderivadanumerica(a) > 0):
            a = np.random.uniform(self.valor_inicial, self.limite)
        
        while (self.primeraderivadanumerica(b) < 0): 
            b = np.random.uniform(self.valor_inicial, self.valor_inicial)
        x1=a
        x2=b
        z = ((x2+x1)/2)
        #print(primeraderivadanumerica(x1,f))
        while(self.primeraderivadanumerica(z) > self.epsilon):
            #print(z)
            if self.primeraderivadanumerica(z) < 0: 
                x1=z
                z=0
                z = int((x2+x1)/2)
            elif self.primeraderivadanumerica(z) > 0: 
                x2=z
                z=0
                z = ((x2+x1)/2)
        
        print("Listo!")
        return x1 , x2

class secante(derivative_methods):
    def __init__(self, x_inicial,limite, f, epsilon):
        super().__init__(x_inicial, f, epsilon)
        self.limite=limite
    def primeraderivadanumerica(self, x_actual):
        delta = 0.0001
        numerador = self.funcion(x_actual + delta) - self.funcion(x_actual - delta) 
        return numerador / (2 * delta)

    def segundaderivadanumerica(self, x_actual):
        delta = 0.0001
        numerado = self.funcion(x_actual + delta) - (2 * self.funcion(x_actual)) + self.funcion(x_actual - delta)
        return numerado / (delta**2)
    def calculozensecante(self,x2,x1):
        numerador=self.primeraderivadanumerica(x2)
        denominador=((self.primeraderivadanumerica(x2) - self.primeraderivadanumerica (x1)))/(x2-x1)
        op=numerador/denominador
        return x2 - op

    def metodosecante(self):
        a = np.random.uniform(self.valor_inicial, self.limite)
        b = np.random.uniform(self.valor_inicial, self.limite)
        while(self.primeraderivadanumerica(a) > 0):
            a = np.random.uniform(self.valor_inicial, self.limite)
        
        while (self.primeraderivadanumerica(b) < 0): 
            b = np.random.uniform(self.valor_inicial, self.valor_inicial)
        x1=a
        x2=b
        z = self.calculozensecante(x2,x1)
        while(self.primeraderivadanumerica(z) > self.epsilon): 
            if self.primeraderivadanumerica(z) < 0: 
                x1=z
                z=0
                z = self.calculozensecante(x2,x1)
            if self.primeraderivadanumerica(z) > 0: 
                x2=z
                z=0
                z = self.calculozensecante(x2,x1)
        return x1 , x2

    
class random_walk(direct_methods):
    def __init__(self, variables, f, epsilon, iter=100):
        super().__init__(variables, f, epsilon, iter)
    
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
        while(self.iteraciones > cont):
            x_nuevo=self.step_calculation(x)
            if self.funcion(x_nuevo)< self.funcion(xmejor):
                xmejor=x_nuevo
            cont+=1
        return xmejor
    
class neldermead(direct_methods):
    def __init__(self, variables,gamma,beta, f, epsilon, iter=100):
        super().__init__(variables, f, epsilon, iter)
        self.variables= np.array(variables)
        self.gamma=gamma
        self.beta=beta
    
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
    def neldermeadmead(self):
        cont = 1
        mov = []
        simplex = self.create_simplex(self.variables)
        mov.append(simplex)
        best, secondworst, worst = self.findpoints(simplex)
        indices = [best, secondworst, worst]
        indices.remove(worst)
        centro = self.xc_calculation(simplex, indices)
        x_r = (2 * centro) - simplex[worst]
        x_new = x_r
        if self.funcion(x_r) < self.funcion(simplex[best]): 
            x_new = ((1 + self.gamma) * centro) - (self.gamma * simplex[worst])
        elif self.funcion(x_r) >= self.funcion(simplex[worst]):
            x_new = ((1 - self.beta) * centro) + (self.beta * simplex[worst])
        elif self.funcion(simplex[secondworst]) < self.funcion(x_r) and self.funcion(x_r) < self.funcion(simplex[worst]):
            x_new = ((1 - self.beta) * centro) - (self.beta * simplex[worst])
        simplex[worst] = x_new
        mov.append(np.copy(simplex))
        stop = self.stopcondition(simplex, centro)
        while stop >= epsilon:
            stop = 0
            best, secondworst, worst = self.findpoints(simplex)
            indices = [best, secondworst, worst]
            indices.remove(worst)
            centro = self.xc_calculation(simplex, indices)
            x_r = (2 * centro) - simplex[worst]
            x_new = x_r
            if self.funcion(x_r) < self.funcion(simplex[best]):
                x_new = ((1 + self.gamma) * centro) - (self.gamma * simplex[worst])
            elif self.funcion(x_r) >= self.funcion(simplex[worst]):
                x_new = ((1 - self.beta) * centro) + (self.beta * simplex[worst])
            elif self.funcion(simplex[secondworst]) < self.funcion(x_r) and self.funcion(x_r) < self.funcion(simplex[worst]):
                x_new = ((1 + self.beta) * centro) - (self.beta * simplex[worst])
            simplex[worst] = x_new
            stop = self.stopcondition(simplex, centro)
            #print(stop)
            #mov.append(np.copy(simplex))
            cont+=1
        return simplex[best]
    
class hooke_jeeves(direct_methods):
    def __init__(self, variables,delta ,f, epsilon, alpha=2):
        super().__init__(variables, f, epsilon)
        self.alpha=alpha
        self.delta=delta
    
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

    def updatedelta(self,delta):
        new_delta = delta / self.alpha
        return new_delta
    def hookejeeves(self):
        cont = 0
        x_inicial = np.array(self.variables)
        delta = np.array(self.delta)
        x_anterior = x_inicial
        x_mejor, flag = self.movexploratory(x_inicial, delta)
        print(x_mejor)
        while np.linalg.norm(delta) > self.epsilon:
            if flag:
                x_point = self.patternmove(x_mejor, x_anterior)
                x_mejor_nuevo, flag = self.movexploratory(x_point, delta)
            else:
                delta = self.updatedelta(delta)
                x_mejor, flag = self.movexploratory(x_mejor, delta)
                x_point = self.patternmove(x_mejor, x_anterior)
                x_mejor_nuevo, flag = self.movexploratory(x_point, delta)
            #Son dos subprocersos
            if self.funcion(x_mejor_nuevo) < self.funcion(x_mejor):
                flag = True
                x_anterior = x_mejor
                x_mejor = x_mejor_nuevo
            else:
                flag = False

            cont += 1
        print("Num de iteraciones {}".format(cont))
        return x_mejor_nuevo

if __name__=="__main__":
    def funcion1(x):
        return (x**2) + (54/x)
    def himmelblau(p):
        return (p[0]**2 + p[1] - 11)**2 + (p[0] + p[1]**2 - 7)**2
    def boothfunction(x):
        return ((x[0] + 2 * (x[1]) - 7) ** 2) + ((2 * x[0]) + x[1] - 5) ** 2
    
    epsilon=0.01
    a=1
    b=10
    x_inicial = 1
    inicial=[-5,-2.5]
    delta=[0.5,0.25]
    gamma = 1.1
    beta= 0.5
    hooke_jeeves_optimizer = hooke_jeeves(variables=inicial, delta=delta, f=boothfunction, epsilon=epsilon)
    
    # Ejecutar el método Hooke-Jeeves
    resultado = hooke_jeeves_optimizer.hookejeeves()
    
    # Imprimir el resultado
    print("Resultado de la optimización:", resultado)