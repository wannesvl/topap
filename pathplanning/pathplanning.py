# TOPAP -- Time optimal path following for differentially flat systems
# Copyright (C) 2013 Wannes Van Loock, KU Leuven. All rights reserved.
#
# TOPAP is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 3 of the License, or (at your option) any later version.
# This software is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA


import casadi as cas
import numpy as np
import warnings
from scipy.misc import factorial
from splines import BSplineBasis
from pathplanning import *


class PathPlanning(PathFollowing):
    def __init__(self, sys):
        super(ConvexCombination, self).__init__(sys)
        self.h = None  # homotopy parameters

    def set_path(self, paths):
        """The path is defined as the convex combination of the paths in paths.

        Args:
            paths (list of lists of SXMatrix): The path is taken as the
            convex combination of the paths in paths.

        Example:
        The path is defined as the convex combination of
        (s, 0.5*s) and (2, 2*s):
            >>> P.set_path([(P.s[0], 0.5 * P.s[0]), [P.s[0], 2 * P.s[0]]])
        """
        l = len(paths)
        self.h = cas.ssym("h", l, self.sys.order + 1)
        self.path[:, 0] = np.sum(cas.SXMatrix(paths) * cas.horzcat([self.h[:, 0]] * len(paths[0])), axis=0)
        dot_s = cas.vertcat([self.s[1:], 0])
        dot_h = cas.horzcat([self.h[:, 1:], cas.SXMatrix.zeros(l, 1)])
        for i in range(1, self.sys.order + 1):  # Chainrule
            self.path[:, i] = (cas.mul(cas.jacobian(self.path[:, i - 1], self.s), dot_s) +
                sum([cas.mul(cas.jacobian(self.path[:, i - 1], self.h[j, :]), dot_h[j, :].trans()) for j in range(l)]) * self.s[1])

    def solve(self):
        """
        define extra variables

        homotopy parameters >= 0!!
        """
        if not self.prob['s']:
            self.set_grid()
        N = self.options['N']
        Nc = self.options['Nc']
        self.prob['vars'] = [cas.ssym("b", N + 1, self.sys.order),
                             cas.ssym("h", Nc, self.h.size1())]
        # Vectorize variables
        V = cas.vertcat([
            cas.vec(self.prob['vars'][0]),
            cas.vec(self.prob['vars'][1])
            ])
        self._make_constraints()
        self._make_objective()
        self._h_init()
        con = cas.SXFunction([V], [self.prob['con'][0]])
        obj = cas.SXFunction([V], [self.prob['obj']])
        if self.options.get('solver') == 'Ipopt':
            solver = cas.IpoptSolver(obj, con)
        else:
            print """Other solver than Ipopt are currently not supported,
            switching to Ipopt"""
            solver = cas.IpoptSolver(obj, con)
        for option, value in self.options.iteritems():
            if solver.hasOption(option):
                solver.setOption(option, value)
        solver.init()
        # Setting constraints
        solver.setInput(cas.vertcat(self.prob['con'][1]), "lbg")
        solver.setInput(cas.vertcat(self.prob['con'][2]), "ubg")
        solver.setInput(
            cas.vertcat([
                [np.inf] * self.sys.order * (N + 1),
                [1] * self.h.size1() * Nc
                ]), "ubx"
            )
        solver.setInput(
            cas.vertcat([
                [0] * (N + 1),
                [-np.inf] * (self.sys.order - 1) * (N + 1),
                [0] * self.h.size1() * Nc
                ]), "lbx"
            )
        solver.solve()
        self.prob['solver'] = solver
        self._get_solution()

    def _h_init(self):
        """Add boundary constraints for the homotopy parameters"""
        # H = self.prob['vars'][1]
        # con = cas.vec(cas.vertcat([cas.vertcat((H[0, :] - H[i, :], H[-1, :] - H[-1 - i, :])) for i in range(1, self.sys.order)]))
        # self.prob['con'][0].append(con)
        # self.prob['con'][1].extend([0] * con.size())
        # self.prob['con'][2].extend([0] * con.size())
        # self.prob['con'][0].append(H[0, 0])
        # self.prob['con'][1].extend([1])
        # self.prob['con'][2].extend([1])
        # self.prob['con'][0].append(H[-1, 0])
        # self.prob['con'][1].extend([0])
        # self.prob['con'][2].extend([0])

    def _make_constraints(self):
        """Parse the constraints and put them in the correct format"""
        N = self.options['N']
        Nc = self.options['Nc']
        b = self.prob['vars'][0]
        H = self.prob['vars'][1]
        # ODEs
        # ~~~~
        con = self._ode(b)
        lb = np.alen(con) * [0]
        ub = np.alen(con) * [0]
        # Convex combination
        # ~~~~~~~~~~~~~~~~~~
        con.append(cas.sumCols(H))
        lb.extend([1] * Nc)
        ub.extend([1] * Nc)
        # Sample constraints
        # ~~~~~~~~~~~~~~~~~~
        S = self.prob['s']
        path, bs = self._make_path()[0:2]
        basisH = self._make_basis()
        B = [np.matrix(basisH(S))]
        # TODO!!! ============================================================
        for i in range(1, self.h.size2()):
            # B.append(np.matrix(basisH.derivative(S, i)))
            Bi, p = basisH.derivative(i)
            B.append(np.matrix(np.dot(Bi(S), p)))
        # ====================================================================
        for f in self.constraints:
            F = cas.substitute(f[0], self.sys.y, path)
            if f[3] is None:
                F = cas.vertcat([cas.substitute(F,
                                 cas.vertcat([
                                    self.s[0],
                                    bs,
                                    cas.vec(self.h)]),
                                 cas.vertcat([
                                    S[j],
                                    cas.vec(b[j, :]),
                                    cas.vec(cas.vertcat([cas.mul(B[i][j, :], H) for i in range(self.h.size2())]).trans())
                                 ])) for j in range(N + 1)])
                Flb = [evalf(cas.SXFunction([self.s], [cas.SXMatrix(f[1])]), s).toArray().ravel() for s in S]
                Fub = [evalf(cas.SXFunction([self.s], [cas.SXMatrix(f[2])]), s).toArray().ravel() for s in S]
                con.append(F)
                lb.extend(Flb)
                ub.extend(Fub)
            else:
                F = cas.vertcat([cas.substitute(F,
                                 cas.vertcat([
                                    self.s[0],
                                    bs,
                                    cas.vec(self.h)]),
                                 cas.vertcat([
                                    S[j],
                                    cas.vec(b[j, :]),
                                    cas.vec(cas.vertcat([cas.mul(B[i][j, :], H) for i in range(self.h.size2())]).trans())
                                 ])) for j in f[3]])
                con.append(F)
                lb.extend([f[1]])
                ub.extend([f[2]])
        self.prob['con'] = [con, lb, ub]

    def _make_basis(self):
        """Return B-spline basis used to evaluate convex conbination function"""
        Nc = self.options['Nc']
        k = self.sys.order + 1
        g = Nc - k - 1  # Number of internal knots
        knots = np.hstack(([0] * k, np.linspace(0, 1, g + 2), [1] * k))
        B = BSplineBasis(knots, k)
        return B

    def _get_solution(self):
        solver = self.prob['solver']
        N = self.options['N']
        Nc = self.options['Nc']
        x_opt = np.array(solver.getOutput("x")).ravel()
        delta = np.diff(self.prob['s'])
        b_opt = np.reshape(x_opt[:self.sys.order * (N + 1)], (N + 1, -1), order='F')
        h_opt = np.reshape(x_opt[self.sys.order * (N + 1):], (Nc, -1), order='F')
        time = np.cumsum(np.hstack([0, 2 * delta / (np.sqrt(b_opt[:-1, 0]) +
                                                 np.sqrt(b_opt[1:, 0]))]))
        # Resample to constant time-grid
        t = np.linspace(time[0], time[-1], self.options['Nt'])
        b_opt = np.array([np.interp(t, time, b) for b in b_opt.T]).T
        # Get s and derivatives from b_opt
        s = np.interp(t, time, self.prob['s'])
        b, Ds = self._make_path()[1:]
        Ds_f = cas.SXFunction([b], [Ds])  # derivatives of s wrt b
        Ds_f.init()
        s_opt = np.vstack((s, np.vstack([evalf(Ds_f, bb).toArray().ravel()
                                          for bb in b_opt]).T)).T
        self.sol['s'] = s_opt
        self.sol['h'] = h_opt
        self.sol['b'] = b_opt
        self.sol['t'] = t
        # Evaluate the states
        basisH = self._make_basis()
        B = [np.dot(basisH(s_opt[:, 0]), h_opt)]
        for i in range(1, self.h.size2()):
            # B.append(np.matrix(np.dot(basisH.derivative(s_opt[:, 0], i), h_opt)))
            Bi, p = basisH.derivative(i)
            B.append(np.matrix(np.dot(np.dot(Bi(s_opt[:, 0]), p), h_opt)))
        f = cas.SXFunction([cas.vertcat([self.s, cas.vec(self.h)])], [cas.substitute(self.sys.x.values(),
                                           self.sys.y, self.path)])
        f_val = np.array([evalf(f, s.T).toArray().ravel() for s in np.hstack((s_opt, np.hstack(B)))])
        self.sol['states'] = dict([(k, f_val[:, i]) for i, k in
                          enumerate(self.sys.x.keys())])
