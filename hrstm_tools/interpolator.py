# @author Hillebrand, Fabian
# @date   2019

import numpy as np

class Interpolator:
    """
    Provides an interpolator for a regular grid with consistent stepsize along 
    an axis.

    The interpolation scheme used is currently piecewise linear polynomials.
    As such, the convergence rate is algebraic with a rate of 2.
    First derivatives are achieved using second order finite differences to not
    stump the convergence rate.

    Pay attention: No care is taken for periodicity or out of bound: Make sure 
                   all points to be interpolated are within the regular grid!
    """

    def __init__(self, x, f):
        self.x = x
        self.f = f
        self.dx = x[0][1]-x[0][0]
        self.dy = x[1][1]-x[1][0]
        self.dz = x[2][1]-x[2][0]
        # For derivative
        self.derF = np.gradient(self.f,self.dx,self.dy,self.dz,
            edge_order=2,axis=(0,1,2))

    ### ------------------------------------------------------------------------
    ### Evaluation functions
    ### ------------------------------------------------------------------------

    def __call__(self, x, y, z):
        """ Evaluates interpolated value at given points. """
        indX = ((x-self.x[0][0])/self.dx).astype(int)
        indY = ((y-self.x[1][0])/self.dy).astype(int)
        indZ = ((z-self.x[2][0])/self.dz).astype(int)

        return ((self.x[0][indX+1]-x)*(                                 \
                  (self.x[1][indY+1]-y)*(                               \
                     (self.x[2][indZ+1]-z)*self.f[indX,indY,indZ]       \
                    +(z-self.x[2][indZ])*self.f[indX,indY,indZ+1])      \
                 +(y-self.x[1][indY])*(                                 \
                     (self.x[2][indZ+1]-z)*self.f[indX,indY+1,indZ]     \
                    +(z-self.x[2][indZ])*self.f[indX,indY+1,indZ+1]))   \
               +(x-self.x[0][indX])*(                                   \
                  (self.x[1][indY+1]-y)*(                               \
                     (self.x[2][indZ+1]-z)*self.f[indX+1,indY,indZ]     \
                    +(z-self.x[2][indZ])*self.f[indX+1,indY,indZ+1])    \
                 +(y-self.x[1][indY])*(                                 \
                     (self.x[2][indZ+1]-z)*self.f[indX+1,indY+1,indZ]   \
                    +(z-self.x[2][indZ])*self.f[indX+1,indY+1,indZ+1])))\
            / (self.dx*self.dy*self.dz)

    def gradient(self, x, y, z, direct):
        """
        Evaluates gradient of interpolation in specified direction 
        (x=1,y=2,z=3).

        Pay attention if units used for grid differ from units for 
        function!
        """
        try:
            tmp = self.derF[direct-1]
        except IndexError:
            raise NotImplementedError( \
                "Gradient in direction {} is not available".format(direct))

        indX = ((x-self.x[0][0])/self.dx).astype(int)
        indY = ((y-self.x[1][0])/self.dy).astype(int)
        indZ = ((z-self.x[2][0])/self.dz).astype(int)

        return ((self.x[0][indX+1]-x)*(                              \
                  (self.x[1][indY+1]-y)*(                            \
                     (self.x[2][indZ+1]-z)*tmp[indX,indY,indZ]       \
                    +(z-self.x[2][indZ])*tmp[indX,indY,indZ+1])      \
                 +(y-self.x[1][indY])*(                              \
                     (self.x[2][indZ+1]-z)*tmp[indX,indY+1,indZ]     \
                    +(z-self.x[2][indZ])*tmp[indX,indY+1,indZ+1]))   \
               +(x-self.x[0][indX])*(                                \
                  (self.x[1][indY+1]-y)*(                            \
                     (self.x[2][indZ+1]-z)*tmp[indX+1,indY,indZ]     \
                    +(z-self.x[2][indZ])*tmp[indX+1,indY,indZ+1])    \
                 +(y-self.x[1][indY])*(                              \
                     (self.x[2][indZ+1]-z)*tmp[indX+1,indY+1,indZ]   \
                    +(z-self.x[2][indZ])*tmp[indX+1,indY+1,indZ+1])))\
           / (self.dx*self.dy*self.dz)
