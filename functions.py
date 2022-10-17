"""
A library of functions
"""
import types
import numpy as np
import matplotlib.pyplot as plt
import numbers
import unittest

class AbstractFunction:
    """
    An abstract function class
    """

    def derivative(self):
        """
        returns another function f' which is the derivative of x
        """
        raise NotImplementedError("derivative")


    def __str__(self):
        return "AbstractFunction"


    def __repr__(self):
        return "AbstractFunction"


    def evaluate(self, x):
        """
        evaluate at x

        assumes x is a numeric value, or numpy array of values
        """
        raise NotImplementedError("evaluate")


    def __call__(self, x):
        """
        if x is another AbstractFunction, return the composition of functions

        if x is a string return a string that uses x as the indeterminate

        otherwise, evaluate function at a point x using evaluate
        """
        if isinstance(x, AbstractFunction):
            return Compose(self, x)
        elif isinstance(x, str):
            return self.__str__().format(x)
        else:
            return self.evaluate(x)


    # the rest of these methods will be implemented when we write the appropriate functions
    def __add__(self, other):
        """
        returns a new function expressing the sum of two functions
        """
        return Sum(self, other)


    def __mul__(self, other):
        """
        returns a new function expressing the product of two functions
        """
        return Product(self, other)


    def __neg__(self):
        return Scale(-1)(self)


    def __truediv__(self, other):
        return self * other**-1


    def __pow__(self, n):
        return Power(n)(self)


    def plot(self, vals=np.linspace(-1,1,100), **kwargs):
        """
        plots function on values
        pass kwargs to plotting function
        """
        plt.plot(vals,self.evaluate(vals),**kwargs)

    def taylor_series(self, x0, deg=5):
        """
        Returns the Taylor series of f centered at x0 truncated to degree k.
        """
        func = self
        result = Constant(func.evaluate(x0))

        for k in range(1,deg+1):
            func = func.derivative()
            coef = func.evaluate(x0)/np.math.factorial(k)
            kterm = Product(Constant(coef),Power(k)(Polynomial(1,-1*x0)))
            result = Sum(result,kterm)

        return result
    

            
    
def newton_root(f, x0, tol=1e-8):

    """
    Returns the root via newton method
    """   
    if not isinstance(f, Symbolic) and isinstance(f, AbstractFunction):
        x = x0
        fx = f.evaluate(x)
        iters = []
        fs = []
        while (abs(fx) > tol):
            fp = f.derivative()
            x = x - fx / fp.evaluate(x)
            fx = f.evaluate(x)
            iters.append(x)
            fs.append(fx)
        return x
        

def newton_extremum(f, x0, tol=1e-8):
       
    fp = f.derivative()
    return newton_root(fp,x0, tol=1e-8)


class Compose(AbstractFunction):
    def __init__(self,f,g):
        self.f = f
        self.g = g

    def __str__(self):
        return "{}".format(self.f(str(self.g)))

    def __repr__(self):
        return "Compose of {}({})".format(self.f,(self.g))

    def derivative(self):
        return Product(self.g.derivative(),Compose(self.f.derivative(),self.g))

    def evaluate(self, x):
        return self.f(self.g(x))

        
class Sum(AbstractFunction):
    """
    sum of functions
    """
    def __init__(self, f, g):
        self.f=f
        self.g=g
    
    def __repr__(self):
        return f"Sum({self.f}, {self.g})"
        
    def __str__(self):
        return "{}+{}".format(self.f,self.g)

    
    def evaluate(self, x):
        return self.f.evaluate(x) + self.g.evaluate(x)
    
    def derivative(self):
        return Sum(self.f.derivative(), self.g.derivative())

        
class Product(AbstractFunction):
    """
    product of functions
    """
    def __init__(self, f, g):
        self.f=f
        self.g=g
    
    def __repr__(self):
        return f"Product({self.f}, {self.g})"
        
    def __str__(self):
        return "{}*{}".format(self.f,self.g)
    
    def evaluate(self, x):
        if isinstance(self.f, int):
            return self.f * self.g.evaluate(x) 
        if isinstance(self.g, int):
            return self.g * self.f.evaluate(x)
        else:
            return self.f.evaluate(x) * self.g.evaluate(x)
    
    def derivative(self):
        return self.f.derivative() * self.g + self.g.derivative() * self.f


class Power(AbstractFunction):

    def __init__(self, n):
        self.n = n

    def __repr__(self):
        return f"Power({self.n}())"

    def __str__(self):
        if(self.n>0):
            return "({{0}})^{}".format(self.n)
        elif(self.n==0):
            return "1"
        else:
            return "({{0}})^({})".format(self.n)

    def evaluate(self, x):
        """
        evaluate polynomial at x
        """
        if isinstance(x, numbers.Number):
            ret = x**self.n
            return ret
        elif isinstance(x, np.ndarray):
            x = np.array(x)
            return x**self.n

    def derivative(self):
        return Product(Constant(self.n),Power(self.n - 1))


class Sin(AbstractFunction):

    def __init__(self, f, *args):
        self.coeff = np.array(list(args)).flatten()
        self.f = f

    def __repr__(self):
        return "Sin()"

    def __str__(self):
        s = "Sin()({{0}})".format(self.f)
        return s

    def evaluate(self, x):
        """
        evaluate polynomial at x
        """
        if isinstance(x, numbers.Number):
            ret = np.sin(x)
            return ret
        elif isinstance(x, np.ndarray):
            x = np.array(x)
            return np.sin(x)

    def derivative(self):
        return Cos()

    
class Cos(AbstractFunction):

    def __init__(self, f, *args):
        self.coeff = np.array(list(args)).flatten()
        self.f = f

    def __repr__(self):
        return "Cos()"

    def __str__(self):
        s = "Cos()({{0}})".format(self.f)
        return s

    def evaluate(self, x):
        """
        evaluate polynomial at x
        """
        if isinstance(x, numbers.Number):
            ret = np.cos(x)
            return ret
        elif isinstance(x, np.ndarray):
            x = np.array(x)
            return np.cos(x)

    def derivative(self):
        return Product(Constant(-1), Sin())
    

class Log(AbstractFunction):

    def __init__(self, f, *args):
        self.coeff = np.array(list(args)).flatten()
        self.f = f
        
    def __repr__(self):
        return "Log()"

    def __str__(self):
        s = "Log()({{0}})".format(self.f)
        return s

    def evaluate(self, x):
        """
        evaluate polynomial at x
        """
        if isinstance(x, numbers.Number):
            ret = np.log(x)
            return ret
        elif isinstance(x, np.ndarray):
            x = np.array(x)
            return np.log(x)

    def derivative(self):
        return Power(-1)


class Exponential(AbstractFunction):

    def __init__(self, f, *args):
        self.coeff = np.array(list(args)).flatten()
        self.f = f

    def __repr__(self):
        return "Exponential()"

    def __str__(self):
        s = "Exponential()({{0}})".format(self.f)
        return s

    def evaluate(self, x):
        """
        evaluate polynomial at x
        """
        if isinstance(x, numbers.Number):
            ret = np.exp(x)
            return ret
        elif isinstance(x, np.ndarray):
            x = np.array(x)
            return np.exp(x)

    def derivative(self):
        return Exponential()
    




class Polynomial(AbstractFunction):
    """
    polynomial c_n x^n + ... + c_1 x + c_0
    """

    def __init__(self, *args):
        """
        Polynomial(c_n ... c_0)

        Creates a polynomial
        c_n x^n + c_{n-1} x^{n-1} + ... + c_0
        """
        self.coeff = np.array(list(args))


    def __repr__(self):
        return "Polynomial{}".format(tuple(self.coeff))


    def __str__(self):
        """
        We'll create a string starting with leading term first

        there are a lot of branch conditions to make everything look pretty
        """
        s = ""
        deg = self.degree()
        for i, c in enumerate(self.coeff):
            if i < deg-1:
                if c == 0:
                    # don't print term at all
                    continue
                elif c == 1:
                    # supress coefficient
                    s = s + "({{0}})^{} + ".format(deg - i)
                else:
                    # print coefficient
                    s = s + "{}({{0}})^{} + ".format(c, deg - i)
            elif i == deg-1:
                # linear term
                if c == 0:
                    continue
                elif c == 1:
                    # suppress coefficient
                    s = s + "{0} + "
                else:
                    s = s + "{}({{0}}) + ".format(c)
            else:
                if c == 0 and len(s) > 0:
                    continue
                else:
                    # constant term
                    s = s + "{}".format(c)

        # handle possible trailing +
        if s[-3:] == " + ":
            s = s[:-3]

        return s


    def evaluate(self, x):
        """
        evaluate polynomial at x
        """
        if isinstance(x, numbers.Number):
            ret = 0
            for k, c in enumerate(reversed(self.coeff)):
                ret = ret + c * x**k
            return ret
        elif isinstance(x, np.ndarray):
            x = np.array(x)
            # use vandermonde matrix
            return np.vander(x, len(self.coeff)).dot(self.coeff)


    def derivative(self):
        if len(self.coeff) == 1:
            return Polynomial(0)
        return Polynomial(*(self.coeff[:-1] * np.array([n+1 for n in reversed(range(self.degree()))])))


    def degree(self):
        return len(self.coeff) - 1


    def __add__(self, other):
        """
        Polynomials are closed under addition - implement special rule
        """
        if isinstance(other, Polynomial):
            # add
            if self.degree() > other.degree():
                coeff = self.coeff
                coeff[-(other.degree() + 1):] += other.coeff
                return Polynomial(*coeff)
            else:
                coeff = other.coeff
                coeff[-(self.degree() + 1):] += self.coeff
                return Polynomial(*coeff)

        else:
            # do default add
            return super().__add__(other)


    def __mul__(self, other):
        """
        Polynomials are clused under multiplication - implement special rule
        """
        if isinstance(other, Polynomial):
            return Polynomial(*np.polymul(self.coeff, other.coeff))
        else:
            return super().__mul__(other)


class Scale(Polynomial):
    """
    scale function a * x
    """

    def __init__(self, a):
        """
        Scale(a)

        Creates a polynomial
        a * x
        """
        super().__init__(a, 0)



class Constant(Polynomial):
    """
    constant function f(x)=c
    """

    def __init__(self, c):
        """
        Constant(c)

        Creates a constant function f(x)=c
        """
        super().__init__(c)

class Symbolic(AbstractFunction):
    def __init__(self, f):
        self.f = f

    def __repr__(self):
        return f"{self.f}"

    def __str__(self):
        return "{}({{0}})".format(self.f)

    def evaluate(self, x):
        return "{}({})".format(self.f,x)
        # return str(self.f)

    def derivative(self):
        return Symbolic("{}'".format(self.f))

class Affine(Polynomial):
    """
    affine function a * x + b
    """
    def __init__(self, a, b):
        super().__init__(a, b)



if __name__ == '__main__':
    sine = Sin(1,2,3,4)
    g = Symbolic('g')
    h = Symbolic('h')
    f = Product(g,Compose(Power(-1),h))
    print(f.derivative()("x"))
    #print(f.derivative())
    for k in [0,1,3,5]:
        taylor = sine.taylor_series(0, deg=k)
        sine.plot(vals=np.linspace(-3, 3, 100))
        taylor.plot(vals=np.linspace(-3, 3, 100))
        plt.xlabel('x')
        plt.ylabel('Function Value')
        plt.legend(['Sine', 'Taylor Series with k = %s' % k])
        plt.title("Sine and Its Taylor Series with k = %s" % k)
        #plt.show()
        #plt.savefig("series-%s.png" % k)
