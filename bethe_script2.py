from triqs.gf import *
from triqs.gf.tools import fit_legendre
import triqs.utility.mpi as mpi
from triqs.operators import *
from triqs_cthyb import Solver
from h5 import *
import numpy as np
import os.path
import sys

from itertools import product as itp

# U given as an argument when calling python
Us = [float(sys.argv[1])]

def mpi_print(arg):
    if mpi.is_master_node():
        print(arg)

# Tail fitting for the Green's function.
# We use it at the first iteration, to fix the number of Legendre coefficient for a given parameter set.
# For the gapped solutions, a large number of coefficients are needed.
# Also, not the same number of coefficients for the diagonal and off-diagonal components.
def fit_G_l(Gt, n_l_e, n_l_o):
    if n_l_e == 0:
        for n_l in range(7, 150, 2):
            G_l_new = fit_legendre(Gt, n_l)
            if abs(G_l_new['nambu'][0, 0].data[n_l-3]) < abs(G_l_new['nambu'][0, 0].data[n_l-1]):
                mpi_print("Optimal number of legendre coefficients for [0, 0] component: " + str(n_l-2))
                n_l_e = n_l - 2
                break
        if n_l_e == 0:
            mpi_print("Couldn't find optimal number of legendre coefficients for [0, 0] component: " + str(n_l))
            n_l_e = n_l

        for n_l in range(7, 150, 2):
            G_l_new = fit_legendre(Gt, n_l)
            if abs(G_l_new['nambu'][1, 1].data[n_l-3]) < abs(G_l_new['nambu'][1, 1].data[n_l-1]):
                mpi_print("Optimal number of legendre coefficients for [1, 1] component: " + str(n_l - 2))
                n_l_e = np.amax([n_l_e, n_l - 2])
                break
        mpi_print("Number of legendre coefficients for diagonal components: " + str(n_l_e))

        for n_l in range(8, 150, 2):
            G_l_new = fit_legendre(Gt, n_l)
            if abs(G_l_new['nambu'][0, 1].data[n_l-3]) < abs(G_l_new['nambu'][0, 1].data[n_l-1]):
                mpi_print("Optimal number of legendre coefficients for [0, 1] component: " + str(n_l - 2))
                n_l_o = n_l - 2
                break
        if n_l_o == 0:
            mpi_print("Couldn't find optimal number of legendre coefficients for [0, 1] component: " + str(n_l))
            n_l_o = n_l

        for n_l in range(8, 150, 2):
            G_l_new = fit_legendre(Gt, n_l)
            if abs(G_l_new['nambu'][1, 0].data[n_l-3]) < abs(G_l_new['nambu'][1, 0].data[n_l-1]):
                mpi_print("Optimal number of legendre coefficients for [1, 0] component: " + str(n_l - 2))
                n_l_o = np.amax([n_l_o, n_l -2])
                break
        mpi_print("Number of legendre coefficients for off-diagonal components: " + str(n_l_o))

    G_l = fit_legendre(Gt, np.amax([n_l_e, n_l_o]))
    G_l['nambu'][:, :].data[:] = 0 * G_l['nambu'][:, :].data[:]

    G_l_new = fit_legendre(Gt, n_l_e)
    G_l['nambu'][0, 0].data[:n_l_e] = G_l_new['nambu'][0, 0].data[:]
    G_l['nambu'][1, 1].data[:n_l_e] = G_l_new['nambu'][1, 1].data[:]

    G_l_new = fit_legendre(Gt, n_l_o)
    G_l['nambu'][0, 1].data[:n_l_o] = G_l_new['nambu'][0, 1].data[:]
    G_l['nambu'][1, 0].data[:n_l_o] = G_l_new['nambu'][1, 0].data[:]

    return G_l, n_l_e, n_l_o

# Different inverse temperatures
# betas = [1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 23, 26, 30, 34, 38, 42, 46, 50, 55, 60, 65, 70, 80, 90, 100, 110, 120, 140, 160, 180, 200]
# betas = [2.25, 2.75, 3.25, 3.5, 3.75, 4.25, 4.5, 4.75, 5.25, 5.5, 5.75]
# betas = [2.5, 6.25, 6.5, 6.75, 7.25, 7.5, 7.75, 8.25, 8.5, 8.75, 9.25, 9.5, 9.75]
betas = [3.5, 3.75, 4.25, 4.5, 4.75, 5.25, 5.5, 5.75]

# Other parameters for the calculation
t = 1
niter = 20
Delta0 = 0.01
p = {'n_warmup_cycles': 10000,
     'n_cycles': int(1e7/mpi.size),
     'random_seed': 123 * mpi.rank + 321}

for U in Us:
    # Half-filling for now
    mu = U/2

    # Interacting Hamiltonian in the Nambu basis
    h_int = U * n('nambu', 0) * c('nambu', 1)*c_dag('nambu', 1)

    for beta in betas:
        mpi_print("U = %.2f, Beta = %i" % (U, beta))
        S = Solver(beta = beta, gf_struct = [('nambu', [0, 1])])

        # First iteration, we start from the SemiCircular density of state of the Bethe lattice.
        # We also add a small pairing field to break the symmetry explicitly.
        if not(os.path.isfile("results/bethe_SC_beta%.2f_U%.2f_Delta%.2f.h5" % (beta, U, Delta0))):
            niter0 = 0

            # Symmetry breaking field
            Sigma = S.G0_iw['nambu'].copy()
            Sigma.zero()
            Sigma[0, 1] << Delta0
            Sigma[1, 0] << Delta0

            # SemiCircular DOS
            S.G0_iw['nambu'][0, 0] << SemiCircular(2*t)
            S.G0_iw['nambu'][1, 1] = S.G0_iw['nambu'][0, 0].copy()
            S.G0_iw['nambu'][1, 1] << -1 * S.G0_iw['nambu'][1, 1].conjugate()

            # Inversion of the Green function
            S.G_iw['nambu'] << inverse(inverse(S.G0_iw['nambu']) - Sigma)
        # We can also start the calculation from a previous iteration.
        else:
            G_iw = S.G_iw.copy()
            niter0 = 0
            if mpi.is_master_node():
                with HDFArchive("results/bethe_SC_beta%.2f_U%.2f_Delta%.2f.h5" % (beta, U, Delta0)) as A:
                    for i in range(niter):
                        if not "G_iw-iter%i" % i in A:
                            niter0 = i
                            G_iw = A["G_iw-iter%i" % (i-1)]
                            break 
            niter0 = mpi.bcast(niter0)
            S.G_iw = mpi.bcast(G_iw)

        # Number of Legendre coefficients will be refreshed
        n_l_e = 0
        n_l_o = 0
        for i in range(niter0, niter):
            mpi_print("########## Beta = %.2f ##### Iteration = %i" % (beta, i))

            # Self-consistency relation
            S.G0_iw['nambu'][0, 0] << iOmega_n + mu - t**2 * S.G_iw['nambu'][0, 0]
            S.G0_iw['nambu'][0, 1] << + t**2 * S.G_iw['nambu'][0, 1]
            S.G0_iw['nambu'][1, 0] << + t**2 * S.G_iw['nambu'][1, 0]
            S.G0_iw['nambu'][1, 1] << iOmega_n - mu - t**2 * S.G_iw['nambu'][1, 1]
            S.G0_iw['nambu'].invert()

            # Save an older Green function for mixing after solve.
            old_G_iw = S.G_iw.copy()
            S.solve(h_int = h_int, **p)
            
            # Legendre tail fitting
            G_l, n_l_e, n_l_o = fit_G_l(S.G_tau, n_l_e, n_l_o)
            S.G_iw << LegendreToMatsubara(G_l)
            # Mixing to avoid density fluctuations
            S.G_iw << 0.7*S.G_iw + 0.3*old_G_iw

            # Calculation of the self-energy
            Sigma = GfImFreq(beta = beta, indices = [0, 1])
            Sigma << inverse(S.G0_iw['nambu']) - inverse(S.G_iw['nambu'])

            # Save data for this iteration in a h5 file
            if mpi.is_master_node():
                with HDFArchive("results/bethe_SC_beta%.2f_U%.2f_Delta%.2f.h5" % (beta, U, Delta0), 'a') as A:
                    A['Solver-iter%i' % i] = S
                    A['G_iw-iter%i' % i] = S.G_iw
                    A['Sigma-iter%i' % i] = Sigma
