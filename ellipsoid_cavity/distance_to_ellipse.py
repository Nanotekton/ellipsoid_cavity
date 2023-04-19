from scipy import optimize
import numpy as np

#based on https://www.geometrictools.com/Documentation/DistancePointEllipseEllipsoid.pdf
def F_generator(y,e):
    nom = y*e
    denom = e*e
    def F(t):
        x = nom/(t+denom)
        return x.dot(x) -1
    return F

def F_prim_generator(y,e):
    nom = -2*y*y*e*e
    denom = e*e
    def F_prim(t):
        x = t+denom
        x = nom/(x*x*x)
        return x.sum()

def F_bis_generator(y,e):
    nom = 6*y*y*e*e
    denom = e*e
    def F_bis(t):
        x = t+ denom
        x = x*x
        x = x*x
        return (nom/x).sum()

def find_root(e, y):
    min_mask = np.where(e==e.min())[0][0]
    t0 = e[min_mask]*y[min_mask] - e[min_mask]**2
    t1 = e*y
    t1 = t1.dot(t1)**0.5 - e[min_mask]**2
    guess = (t0+t1)/2
    try:
        F = F_generator(y,e)
        t = optimize.root_scalar(F, x0=guess, bracket=(t0,t1)).root
    except:
        print(t0, t1)
        print(F(t0), F(t1))
        raise
    return t

def DistToEllipse(e, y):
    assert e.shape==(2,) and y.shape==(2,)
    assert (e==list(reversed(sorted(e)))).all()
    assert all(y>=0) and all(e>0)
    e0, e1 = e
    y0, y1 = y
    compute = False
    if y1>0:
        if y0>0:
            t = find_root(e, y)
            x0 = e0*e0*y0/(t + e0*e0)
            x1 = e1*e1*y1/(t + e1*e1)
        else:
            x0,x1 = 0, e1
    else:
        c = (e0*e0-e1*e1)/e0
        if y0<c:
            x0 = e0*y0/c
            x1 = x0/e0
            x1 = e1*np.sqrt(1 - x1*x1)
        else:
            x0,  x1 = e0, 0

    d =  np.array([x0, x1])
    return d

def DistToEllipsoid(e, y):
    assert e.shape==(3,) and y.shape==(3,)
    assert (e==list(reversed(sorted(e)))).all(), f'{e}'
    assert all(y>=0) and all(e>0), f'y={y.round(3)} e={e.round(3)}'
    e0, e1, e2 = e
    y0, y1, y2 = y

    if all(y==0):
        x = np.where(e==e.min(), e.min(), 0)
        return x
    
    x = np.zeros(3)
    if y2>0:
        if y1>0:
            if y0>0:
                t = find_root(e,y)
                x = e*e
                x = x*y/(t + x)
            else: #y0 == 0
                x[1:] = DistToEllipse(e[1:], y[1:])
        else: #y1 == 0
            x[2] = e2
            if y0 > 0:
                idx = [0,2]
                x[idx] = DistToEllipse(e[idx], y[idx])
    else: #y2=0
        denom0 = e0*e0 - e2*e2
        denom1 = e1*e1 - e2*e2
        numer0 = e0*y0
        numer1 = e1*y1
        computed = False
        if (numer0 < denom0) and (numer1<denom1):
            xde0 = numer0/denom0
            xde1 = numer1/denom1
            xde0sqr = xde0*xde0
            xde1sqr = xde1*xde1
            discr = 1 - xde0sqr - xde1sqr
            if  discr>0:
                x = e*np.array([xde0, xde1, np.sqrt(discr)])
                computed = True
        if not computed:
            idx = [0,1]
            x[idx] = DistToEllipse(e[idx], y[idx])
    return x
