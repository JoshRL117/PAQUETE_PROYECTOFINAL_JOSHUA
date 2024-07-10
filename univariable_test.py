import numpy as np
class optimizador:
    def __init__(self,variables,epsilon,f,iter=100):
        self.variables =np.array(variables,dtype=float)
        self.epsilon = epsilon
        self.funcion = f
        self.iteracion = iter
    
class optimizador_univariable_by_regions_elimination(optimizador):
    def __init__(self,a, b,epsilon, f, iter=100):
        super().__init__(epsilon, f, iter)
        self.a=a
        self.b=b
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
        xm=(self.a+self.b)/2
        l=self.b-self.a
        x1=a + (l/4)
        x2=b - (l/4)
        a,b=self.findregions(self.a,self.b,x1,x2)
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
    def fibonacci_iterativo(self):
        fibonaccie = [0, 1]
        for i in range(2, self.iteracion*10):
            fibonaccie.append(fibonaccie[i-1] + fibonaccie[i-2])
        return fibonaccie

    def calculo_lk(self,fibonacci,k):
        indice1=self.iteracion - (k + 1)
        indice2= self.iteracion + 1
        return fibonacci[indice1]/ fibonacci[indice2]

    def fibonaccisearch(self):
        l=b-a
        seriefibonacci=self.fibonacci_iterativo()
        #calculo de lk
        k=2
        lk=self.calculo_lk(seriefibonacci,k)
        a,b=self.a,self.b
        x1=a+lk
        x2=b-lk
        while k != self.iteracion:
            if k % 2 == 0:
                evalx1=self.funcion(x1)
                a,b=self.findregions(a,b,evalx1,x2)
                #print(" Valor actual de a y b = {} , {}".format(a,b))
            else:
                evalx2=self.funcion(x2)
                a,b=self.findregions(a,b,x1,evalx2)
                #print(" Valor actual de a y b = {} , {}".format(a,b))
            k+=1
        return a,b
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
    def w_to_x(self, a, b, w: float) -> float:
        return w * (b - a) + a
    
    def busquedadorada(self) -> float:
        PHI = (1 + np.sqrt(5)) / 2 - 1
        a, b = self.a,self.b
        aw, bw = 0, 1
        Lw = 1
        k = 1
        while Lw > self.epsilon:
            w2 = aw + PHI * Lw
            w1 = bw - PHI * Lw
            fx1 = self.funcion(self.w_to_x(aw, bw, w1))
            fx2 = self.funcion(self.w_to_x(aw, bw, w2))
            aw, bw = self.findregions_golden(fx1, fx2, w1, w2, aw, bw)
            k += 1
            Lw = bw - aw
        t=(bw * (b - a) + a)
        t2=(aw * (b - a) + a)
        return (t2 + t)/2