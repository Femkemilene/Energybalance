import numpy as np
from scipy import sparse

def ChangCooper(points, nx, dx, drift, Q):
    """For a constant diagonal diffusion, points is meshgrid of points, nx is 
    the amount of points per dimension, dx is the grid distance in every
    dimension, drift is vector function with argument location vector, Q is
    the diagonal diffusion matrix
    Description of the method can be found in Chang & Cooper, 1970: A 
    Practical Difference Scheme for Fokker-Planck Equations. 
    Implementation by Alexis Tantet"""
    (dim, N) = points.shape #N has length nx[0]*nx[1]*(..)
    rows = []
    cols = []
    data = []

    for k in np.arange(N):
        #loop over all points
        j = np.array(np.unravel_index(k, nx)) #flat index k, shape nx, j is coordinateS
        pj = points[:, k] #take coordinates corresponding to point k

        for d in np.arange(dim):
            #Loop over all dimensions
            
            #the p stands for plus, the m minus          
            h = dx[d]
            # Get indices +1 and -1
            jp1 = j.copy()
            jp1[d] += 1
            jm1 = j.copy()
            jm1[d] -= 1
                
            # Get points +1/2 and -1/2
            pjp = pj.copy()
            pjp[d] += h / 2.
            pjm = pj.copy()
            pjm[d] -= h / 2.

            # Get fields
            Bjp = - drift(pjp)[d]
            Bjm = - drift(pjm)[d]
            Cjp = Q[d, d] / 2.
            Cjm = Q[d, d] / 2.
            
            # Get convex combination weights
            wj = h * Bjp / Cjp
            if np.isposinf(wj):
                deltaj = 0.
            if np.isneginf(wj):
                deltaj = 1.
            elif np.abs(wj) < 1.e-8:
                deltaj = 1./2
            else:
                deltaj = 1. / wj - 1. / (np.exp(wj) - 1)

            wjm1 = h * Bjm / Cjm
            if np.isposinf(wjm1):
                deltajm1 = 0.
            if np.isneginf(wjm1):
                deltajm1 = 1.
            elif np.abs(wjm1) < 1.e-8:
                deltajm1 = 1./2
            else:
                deltajm1 = 1. / wjm1 - 1. / (np.exp(wjm1) - 1)

            # Do not devide by step since we directly do the matrix product
            #Left boundary
            if j[d] == 0:
                kp1 = np.ravel_multi_index(jp1, nx)
                rows.append(k)
                cols.append(k)
                data.append(-(Cjp / h - deltaj * Bjp) / h)
                rows.append(k)
                cols.append(kp1)
                data.append(((1. - deltaj) * Bjp + Cjp / h) / h)
            #Right boundary
            elif j[d] + 1 == nx[d]:
                km1 = np.ravel_multi_index(jm1, nx)
                rows.append(k)
                cols.append(km1)
                data.append((Cjm / h - deltajm1 * Bjm) / h)
                rows.append(k)
                cols.append(k)
                data.append(-(Cjm / h + (1 - deltajm1) * Bjm) / h)
            #The center
            else:
                km1 = np.ravel_multi_index(jm1, nx)
                kp1 = np.ravel_multi_index(jp1, nx)
                rows.append(k)
                cols.append(km1)
                data.append((Cjm / h - deltajm1 * Bjm) / h)
                rows.append(k)
                cols.append(k)
                data.append(-((Cjp + Cjm) / h \
                              + (1 - deltajm1) * Bjm \
                              - deltaj * Bjp) / h)
                rows.append(k)
                cols.append(kp1)
                data.append(((1. - deltaj) * Bjp + Cjp / h) / h)
                
    # Get CRS matrix   
    FPO = sparse.csr_matrix((data, (rows, cols)), shape=(N, N))
    return FPO
