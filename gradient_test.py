import numpy as np 

class optimizador:
    def __init__(self,variables,epsilon,f,iter=100):
        self.variables =np.array(variables,dtype=float)
        self.epsilon = epsilon
        self.funcion = f
        self.iteracion = iter

class optimizador_univariable(optimizador):
    def __init__(self, variables, epsilon, f, iter=100):
        super().__init__(variables, epsilon, f, iter)
        self.puntoinicial = float(variables[0])
    
    
class optimizador_gradiente(optimizador):
    def __init__(self, variables, epsilon, f, iter=100):
        super().__init__(variables, epsilon, f, iter)
        self.gradiente = []
    def testalpha(self, alfa):
        return self.funcion(self.variables - (alfa * np.array(self.gradiente)))
    
    def w_to_x(self, a, b, w: float) -> float:
        return w * (b - a) + a

    def busquedadorada(self,f) -> float:
        PHI = (1 + np.sqrt(5)) / 2 - 1
        a, b = 0,1
        aw, bw = 0, 1
        Lw = 1
        k = 1
        while Lw > self.epsilon:
            w2 = aw + PHI * Lw
            w1 = bw - PHI * Lw
            fx1 = f(self.w_to_x(aw, bw, w1))
            fx2 = f(self.w_to_x(aw, bw, w2))
            aw, bw = self.findregions_golden(fx1, fx2, w1, w2, aw, bw)
            k += 1
            Lw = bw - aw
        t=(bw * (b - a) + a)
        t2=(aw * (b - a) + a)
        return (t2 + t)/2
    def primeraderivadanumerica(self, x_actual, f):
        delta = 0.0001
        numerador = f([x_actual + delta]) - f([x_actual - delta])
        return numerador / (2 * delta)

    def segundaderivadanumerica(self, x_actual, f):
        delta = 0.0001
        numerador = f([x_actual + delta]) - (2 * f([x_actual])) + f([x_actual - delta])
        return numerador / (delta**2)

    def newton_raphson(self):
        cont = 0
        x_actual = self.puntoinicial
        xderiv = self.primeraderivadanumerica(x_actual, self.testalpha)
        xderiv2 = self.segundaderivadanumerica(x_actual, self.testalpha)
        xsig = x_actual - (xderiv / xderiv2)
        x_actual = xsig
        while abs(self.primeraderivadanumerica(xsig, self.testalpha)) > self.epsilon:
            xderiv = self.primeraderivadanumerica(x_actual, self.testalpha)
            xderiv2 = self.segundaderivadanumerica(x_actual, self.testalpha)
            xsig = x_actual - (xderiv / xderiv2)
            x_actual = xsig
            cont += 1
        return xsig

    def biseccionmethod(self):
        a = np.random.uniform(self.puntoinicial, max(self.variables))
        b = np.random.uniform(self.puntoinicial, max(self.variables))
        while self.primeraderivadanumerica(a, self.testalpha) > 0:
            a = np.random.uniform(self.puntoinicial, max(self.variables))

        while self.primeraderivadanumerica(b, self.testalpha) < 0:
            b = np.random.uniform(self.puntoinicial, max(self.variables))
        x1 = a
        x2 = b
        z = (x2 + x1) / 2
        while self.primeraderivadanumerica(z, self.testalpha) > self.epsilon:
            if self.primeraderivadanumerica(z, self.testalpha) < 0:
                x1 = z
                z = (x2 + x1) / 2
            elif self.primeraderivadanumerica(z, self.testalpha) > 0:
                x2 = z
                z = (x2 + x1) / 2
        return x1, x2

    def calculozensecante(self, x2, x1, f):
        numerador = self.primeraderivadanumerica(x2, f)
        denominador = (self.primeraderivadanumerica(x2, f) - self.primeraderivadanumerica(x1, f)) / (x2 - x1)
        op = numerador / denominador
        return x2 - op

    def metodosecante(self):
        a = np.random.uniform(self.puntoinicial, max(self.variables))
        b = np.random.uniform(self.puntoinicial, max(self.variables))
        while self.primeraderivadanumerica(a, self.testalpha) > 0:
            a = np.random.uniform(self.puntoinicial, max(self.variables))

        while self.primeraderivadanumerica(b, self.testalpha) < 0:
            b = np.random.uniform(self.puntoinicial, max(self.variables))
        x1 = a
        x2 = b
        z = self.calculozensecante(x2, x1, self.testalpha)
        while self.primeraderivadanumerica(z, self.testalpha) > self.epsilon:
            if self.primeraderivadanumerica(z, self.testalpha) < 0:
                x1 = z
                z = self.calculozensecante(x2, x1, self.testalpha)
            if self.primeraderivadanumerica(z, self.testalpha) > 0:
                x2 = z
                z = self.calculozensecante(x2, x1, self.testalpha)
        return x1, x2

    def findregions(self, rangomin, rangomax, x1, x2):
        if self.testalpha(x1) > self.testalpha(x2):
            rangomin = rangomin
            rangomax = x2
        elif self.testalpha(x1) < self.testalpha(x2):
            rangomin = x1
            rangomax = rangomax
        elif self.testalpha(x1) == self.testalpha(x2):
            rangomin = x1
            rangomax = x2
        return rangomin, rangomax

    def findregions_golden(self, fx1, fx2, rangomin, rangomax, x1, x2):
        if fx1 > fx2:
            rangomin = rangomin
            rangomax = x2
        elif fx1 < fx2:
            rangomin = x1
            rangomax = rangomax
        elif fx1 == fx2:
            rangomin = x1
            rangomax = x2
        return rangomin, rangomax

    def intervalstep3(self, b, x1, xm):
        if self.testalpha(x1) < self.testalpha(xm):
            b = xm
            xm = x1
            return b, xm, True
        else:
            return b, xm, False

    def intervalstep4(self, a, x2, xm):
        if self.testalpha(x2) < self.testalpha(xm):
            a = xm
            xm = x2
            return a, xm, True
        else:
            return a, xm, False

    def intervalstep5(self, b, a):
        l = b - a
        if abs(l) < self.epsilon:
            return False
        else:
            return True

    def intervalhalvingmethod(self):
        a = 0
        b = 1
        xm = (a + b) / 2
        l = b - a
        x1 = a + (l / 4)
        x2 = b - (l / 4)
        a, b = self.findregions(a, b, x1, x2)
        endflag = self.intervalstep5(a, b)
        l = b - a
        while endflag:
            x1 = a + (l / 4)
            x2 = b - l / 4
            b, xm, flag3 = self.intervalstep3(b, x1, xm)
            a, xm, flag4 = self.intervalstep4(a, x2, xm)
            if flag3 == True:
                endflag = self.intervalstep5(a, b)
            elif flag3 == False:
                a, xm, flag4 = self.intervalstep4(a, x2, xm)
            if flag4 == True:
                endflag = self.intervalstep5(a, b)
            elif flag4 == False:
                a = x1
                b = x2
                endflag = self.intervalstep5(a, b)
        return xm

    def fibonacci_iterativo(self, n):
        fibonaccie = [0, 1]
        for i in range(2, n):
            fibonaccie.append(fibonaccie[i - 1] + fibonaccie[i - 2])
        return fibonaccie

    def calculo_lk(self, fibonacci, n, k):
        indice1 = n - (k + 1)
        indice2 = n + 1
        return fibonacci[indice1] / fibonacci[indice2]

    def fibonaccisearch(self):
        a_inicial = 0
        b_inicial = 1
        l = b_inicial - a_inicial
        seriefibonacci = self.fibonacci_iterativo(self.iteracion * 10)
        k = 2
        lk = self.calculo_lk(seriefibonacci, self.iteracion, k)
        x1 = a_inicial + lk
        x2 = b_inicial - lk
        a = a_inicial
        b = b_inicial
        while k != self.iteracion:
            if k % 2 == 0:
                evalx1 = self.testalpha(x1)
                a, b = self.findregions(a, b, evalx1, x2)
            else:
                evalx2 = self.testalpha(x2)
                a, b = self.findregions(a, b, x1, evalx2)
            k += 1
        return a, b

