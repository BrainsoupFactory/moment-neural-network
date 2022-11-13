#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2020 Yang Qi, ISTBI, Fudan University China
import warnings
import numpy as np
from scipy.integrate import quad
from scipy.special import erfcx, erfi, erfc, dawsn

warnings.filterwarnings('ignore')


class Coeffs():
    def __init__(self):
        self.asym_neginf = 0
        self.asym_posinf = 0
        self.taylor = 0
        self.int_asym_neginf = 0


class Dawson1:
    def __init__(self):
        '''1st order Dawson function and its integral'''
        self.coef = Coeffs()
        self.coef.div = 4
        self.coef.deg = 8
        self.coef.cheb_xmin_for_G = -6.0
        self.coef.cheb_G_neg = Chebyshev.chebfit_no_transform(self.int_brute_force, xmin=self.coef.cheb_xmin_for_G,
                                                              xmax=0, num_subdiv=self.coef.div,
                                                              degree_cheb=self.coef.deg)
        self.coef.cheb_g_neg = Chebyshev.chebfit_neg(self.dawson1, num_subdiv=self.coef.div, degree_cheb=self.coef.deg)

        return

    def dawson1(self, x):
        '''Compute Dawson function with existing library'''
        y = erfcx(-x) * np.sqrt(np.pi) / 2
        return y

    def dawson1_custom(self, x):
        '''Compute Dawson function with custom implementation'''
        region1 = -np.abs(x) < self.coef.cheb_xmin_for_G
        region2 = ~region1
        region2_pos = x > 0

        y = np.zeros(x.size)

        y[region1] = self.asym_neginf(-np.abs(x[region1]))
        y[region2] = Chebyshev.chebval_neg(-np.abs(x[region2]), self.coef.cheb_g_neg, num_subdiv=self.coef.div,
                                           degree_cheb=self.coef.deg)
        y[region2_pos] = np.sqrt(np.pi) * np.exp(x[region2_pos] * x[region2_pos]) - y[region2_pos]
        return y

    def asym_neginf(self, x):
        '''Compute asymptotic expansion of the indefinite integral of g(x) for x<<-1
        Use recurrence relation so it only contains multiplication and addition.
        a(n+1)/a(n) = -(2n+1)/(2x^2)
        '''
        a = -0.5 / x  # first term
        h = a.copy()
        for n in range(5):
            a = -a * 0.5 * (2 * n + 1) / (x * x)
            h += a

        return h

    def int_fast(self, x):
        '''fast approximation'''

        region1 = -np.abs(x) < self.coef.cheb_xmin_for_G

        region2 = ~region1
        region2_pos = x > 0

        y = np.zeros(x.size)

        y[region1] = self.int_asym_neginf(-np.abs(x[region1]))
        y[region2] = Chebyshev.chebval_no_transform(-np.abs(x[region2]), self.coef.cheb_G_neg,
                                                    xmin=self.coef.cheb_xmin_for_G, xmax=0, num_subdiv=self.coef.div,
                                                    degree_cheb=self.coef.deg)
        y[region2_pos] += np.pi / 2 * erfi(x[region2_pos])

        return y

    def int_asym_neginf(self, x):
        '''Compute asymptotic expansion of the indefinite integral of G(x) for x<<-1'''
        A = [-1 / 8, 3 / 32, -5 / 32]
        # h = 0.25 * (-np.euler_gamma - np.log(4)) - 0.5 * np.log(-x)  # - 0.25*np.real(np.log(-x*x+0j)), just keep the real part
        h = -0.25 * np.euler_gamma - 0.5 * np.log(-2 * x)
        k = 2
        for a in A:
            h += a * np.power(x, -k)
            k += 2
        return h

    def int_brute_force(self, X):
        '''2nd order Dawson function (direct integration)'''
        q = np.zeros(X.size)
        i = 0
        for x in X:
            q[i], _ = quad(lambda x: erfcx(-x), 0, x)
            i += 1
        q = q * np.sqrt(np.pi) / 2
        return q


class Dawson2:
    def __init__(self, N=30):
        '''Provide 2nd order Dawson function and their integrals'''
        self.dawson1 = Dawson1().dawson1
        self.dawson1_int = Dawson1().int_fast
        self.N = N  # truncation
        self.coef = Coeffs()
        self.coef.asym_neginf = np.array(
            [-1 / 8, 5 / 16, -1, 65 / 16, -2589 / 128, 30669 / 256, -52779 / 64, 414585 / 64,
             -117193185 / 2048, 2300964525 / 4096, -6214740525 / 1024, 293158982025 / 4096])
        self.coef.div = 4
        self.coef.deg = 8
        self.coef.div_pos = 6
        self.coef.deg_pos = 8
        self.coef.cheb_xmas_for_H = 4.5
        self.coef.cheb_neg = Chebyshev.chebfit_neg(self.brute_force, num_subdiv=self.coef.div,
                                                   degree_cheb=self.coef.deg)
        self.coef.cheb_H_neg = Chebyshev.chebfit_neg(self.int_exact, num_subdiv=self.coef.div,
                                                     degree_cheb=self.coef.deg)
        self.coef.cheb_H_pos = Chebyshev.chebfit_no_transform(lambda x: self.int_exact(x) / np.exp(2 * x * x), xmin=0,
                                                              xmax=self.coef.cheb_xmas_for_H,
                                                              num_subdiv=self.coef.div_pos,
                                                              degree_cheb=self.coef.deg_pos)

    def dawson2(self, x):
        """
		2nd order Dawson function (fast approximation) with transformation and subdivision
		"""
        region1 = x < -10.0
        region3 = x > 10.0
        region2 = ~(region1 | region3)
        region2_pos = region2 & (x > 0)

        y = np.zeros(x.size)
        y[region1] = self.asym_neginf(x[region1])
        y[region3] = self.asym_posinf(x[region3])
        y[region2] = Chebyshev.chebval_neg(-np.abs(x[region2]), self.coef.cheb_neg, num_subdiv=self.coef.div,
                                           degree_cheb=self.coef.deg)

        x_pos = x[region2_pos]
        G = self.dawson1_int(-x_pos)
        y[region2_pos] = np.sqrt(np.pi) * np.exp(x_pos * x_pos) * \
                         (0.5 * np.log(2) + 2 * G + np.pi / 2 * erfi(x_pos)) \
                         - y[region2_pos]

        return y

    def int_fast(self, x):
        """2nd order Dawson function (fast approximation)"""
        region1 = x < -10.0
        region3 = x > self.coef.cheb_xmas_for_H
        region2_neg = ~(region1 | region3) & (x <= 0)
        region2_pos = ~(region1 | region3) & (x > 0)

        y = np.zeros(x.size)
        y[region1] = self.int_asym_neginf(x[region1])
        y[region3] = self.int_asym_posinf(x[region3])
        y[region2_neg] = Chebyshev.chebval_neg(x[region2_neg], self.coef.cheb_H_neg, num_subdiv=self.coef.div,
                                               degree_cheb=self.coef.deg)
        y[region2_pos] = np.exp(2 * x[region2_pos] * x[region2_pos]) * Chebyshev.chebval_no_transform(x[region2_pos],
                                                                                                      self.coef.cheb_H_pos,
                                                                                                      xmin=0,
                                                                                                      xmax=self.coef.cheb_xmas_for_H,
                                                                                                      num_subdiv=self.coef.div_pos,
                                                                                                      degree_cheb=self.coef.deg_pos)
        return y

    def int_brute_force(self, X):
        '''Integral of the 2nd order Dawson function (direct integration)'''
        q = np.zeros(X.size)
        i = 0
        fun = lambda x: quad(lambda y: np.exp((x + y) * (x - y)) * (self.dawson1(y) ** 2), -np.inf, x)[0]
        for x in X:
            q[i], _ = quad(fun, -np.inf, x)
            i += 1
        return q

    def int_exact(self, X):
        '''Integral of the 2nd order Dawson function (with a change of order of integration)'''
        q = np.zeros(X.size)
        i = 0
        fun1 = lambda x: np.power(erfcx(-x), 2) * dawsn(x)
        fun2 = lambda x: np.exp(-x * x) * np.power(erfcx(-x), 2)
        for x in X:
            if x < -25:  # == -np.inf:
                q[i] = self.int_asym_neginf(x)
            else:
                y1, _ = quad(fun1, -np.inf, x)
                y2, _ = quad(fun2, -np.inf, x)
                q[i] = -np.pi / 4 * y1 + np.power(np.sqrt(np.pi) / 2, 3) * erfi(x) * y2
            i += 1
        return q

    def brute_force(self, X):
        '''2nd order Dawson function (direct integration)'''
        q = np.zeros(X.size)
        i = 0
        for x in X:
            if x < -1e100:  # @== -np.inf:
                q[i] = 0
            else:
                q[i], _ = quad(lambda y: np.exp((x + y) * (x - y)) * (self.dawson1(y) ** 2), -np.inf, x)
            i += 1
        return q

    def asym_neginf(self, x, N=7):
        '''Asymptotic expansion of H(x) as x-->-Inf. Works well for x<<-3'''
        # WARNING: truncate the expansion at N=7 is good. Larger truncation inreases error so don't change it.
        # Continued fraction doesn't seem to make a big difference on modern hardware.
        h = 0
        for k in range(N):
            h += np.power(x, -3 - 2 * k) * self.coef.asym_neginf[k]
        return h

    def asym_posinf(self, x):
        '''Asymptotic expansion of H(x) as x-->+Inf.'''
        h = np.power(np.sqrt(np.pi) / 2, 3) * np.exp(x * x)
        h *= np.power(erfc(-x), 2) * erfi(x)
        return h

    def int_asym_neginf(self, x, N=7):
        '''Evaluate integral of the 2nd order Dawson function with asymptotic expansion. Works well for x<<-3 '''
        h = 0
        for k in range(N):
            h += np.power(x, -2 - 2 * k) * self.coef.asym_neginf[k] / (-2 - 2 * k)
        return h

    def int_asym_posinf(self, x):

        E1 = erfi(x)
        E2 = np.power(erfc(-x), 2)
        a = np.pi ** 2 / 32
        H = a * (E1 - 1) * E1 * E2

        return H


class Chebyshev():
    @staticmethod
    def chebfit_neg(target_fun, num_subdiv=50, degree_cheb=3, a=4, alpha=1):
        '''Fit a target function on (-inf,0] with Chebyshev polynomial'''
        N = num_subdiv  # number of subdivisions
        K = degree_cheb  # degree of Chebyshev polynomial
        delta_x = 1 / N
        P = np.zeros((N, K + 1))
        for i in range(N):
            xx = np.linspace(0, delta_x, 11) + i * delta_x
            y = -np.power((1 / xx - 1) * a, 1 / alpha)
            h = target_fun(y)
            c = np.polynomial.chebyshev.chebfit(xx, h, K)
            P[i, :] = c
        return P

    @staticmethod
    def chebval_neg(x, P, num_subdiv=50, degree_cheb=3, a=4, alpha=1):
        '''Evaluate a function on (-inf,0] with Chebyshev polynomial'''
        N = num_subdiv  # number of subdivisions
        K = degree_cheb  # degree of Chebyshev polynomial
        delta_x = 1 / N

        # transform the input variable from (-inf,0) to (0,1)
        x = a / (a + np.power(np.abs(x), alpha))

        # Evaluate Chebyshev polynomial for each subinterval
        h_fitted = np.zeros(x.size)
        for i in range(N):
            indx = np.logical_and(x > delta_x * i, x <= delta_x * (i + 1))
            h_fitted[indx] = np.polynomial.chebyshev.chebval(x[indx], P[i, :])
        return h_fitted

    @staticmethod
    def chebfit_no_transform(target_fun, xmin=0, xmax=1, num_subdiv=50, degree_cheb=3):
        '''Fit a target function on [xmin,xmax] with Chebyshev polynomial without transformation'''
        N = num_subdiv  # number of subdivisions
        K = degree_cheb  # degree of Chebyshev polynomial
        delta_x = (xmax - xmin) / N
        P = np.zeros((N, K + 1))
        for i in range(N):
            xx = np.linspace(0, delta_x, 11) + i * delta_x + xmin
            h = target_fun(xx)
            c = np.polynomial.chebyshev.chebfit(xx, h, K)
            P[i, :] = c
        return P

    @staticmethod
    def chebval_no_transform(x, P, xmin=0, xmax=1, num_subdiv=50, degree_cheb=3):
        '''Evaluate a function on [xmin,xmax] with Chebyshev polynomial without transformation'''
        N = num_subdiv  # number of subdivisions
        K = degree_cheb  # degree of Chebyshev polynomial
        delta_x = (xmax - xmin) / N
        # Evaluate Chebyshev polynomial for each subinterval
        h_fitted = np.zeros(x.size)
        for i in range(N):
            indx = np.logical_and(x > xmin + delta_x * i, x <= xmin + delta_x * (i + 1))
            h_fitted[indx] = np.polynomial.chebyshev.chebval(x[indx], P[i, :])
        return h_fitted
