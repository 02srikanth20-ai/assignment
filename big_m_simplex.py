"""
BIG-M SIMPLEX METHOD
=====================
Case study : A furniture manufacturer's diet-style / resource-mix LPP
             (a classic textbook problem with <=, >=, and = constraints,
             so that slack, surplus AND artificial variables are all
             required -- this is exactly the situation the Big-M method
             is built for.)

Problem (Minimization) - after Taha, "Operations Research: An
Introduction" (a widely used textbook example):

    Minimize   Z = 4 x1 + x2

    subject to
        3 x1 +   x2  =  3        ... (equality  -> needs artificial variable)
        4 x1 + 3 x2 >=  6        ... (>=        -> needs surplus + artificial)
          x1 + 2 x2 <=  4        ... (<=        -> needs slack)
        x1, x2 >= 0

Standard form after adding slack (s), surplus (su) and artificial (a)
variables:

    3 x1 +   x2                + a1                 = 3
    4 x1 + 3 x2       - su1          + a2            = 6
      x1 + 2 x2  + s1                                = 4

Big-M objective (minimisation):
    Min Z = 4x1 + x2 + 0.s1 + 0.su1 + M.a1 + M.a2

This program implements a completely generic Big-M Simplex engine
(any number of variables / <=, >=, = constraints) and then uses it to
solve the case study above, printing every simplex iteration.

Author : (generated for coursework demonstration)
"""

from fractions import Fraction
from typing import List, Tuple

BIG_M_SYMBOL = "M"


# ---------------------------------------------------------------------
#  Helper class to do arithmetic on (M-part, constant-part) pairs.
#  Every cost / Cj-Zj value is represented as   m*M + k   with m, k
#  rational numbers.  This lets us treat M as "an arbitrarily large
#  positive number" symbolically instead of plugging in a fixed huge
#  number (which can cause numerical trouble).
# ---------------------------------------------------------------------
class MVal:
    __slots__ = ("m", "k")

    def __init__(self, m=0, k=0):
        self.m = Fraction(m)
        self.k = Fraction(k)

    def __add__(self, other):
        return MVal(self.m + other.m, self.k + other.k)

    def __sub__(self, other):
        return MVal(self.m - other.m, self.k - other.k)

    def __mul__(self, scalar):
        scalar = Fraction(scalar)
        return MVal(self.m * scalar, self.k * scalar)

    __rmul__ = __mul__

    def is_negative(self):
        # M is understood to be an arbitrarily large POSITIVE number,
        # so the sign of (m, k) is decided by m first, then k.
        if self.m != 0:
            return self.m < 0
        return self.k < 0

    def __lt__(self, other):
        d = self - other
        if d.m != 0:
            return d.m < 0
        return d.k < 0

    def __eq__(self, other):
        return self.m == other.m and self.k == other.k

    def __repr__(self):
        if self.m == 0:
            return f"{float(self.k):.3f}"
        if self.k == 0:
            return f"{self.m}{BIG_M_SYMBOL}"
        sign = "+" if self.k > 0 else "-"
        return f"{self.m}{BIG_M_SYMBOL} {sign} {float(abs(self.k)):.3f}"


# ---------------------------------------------------------------------
#  Generic Big-M Simplex engine
# ---------------------------------------------------------------------
class BigMSimplex:
    """
    c          : list[float]   objective coefficients of the ORIGINAL
                                decision variables (already in MIN form)
    A          : list[list]    constraint coefficient matrix (m x n)
    relations  : list[str]     one of '<=', '>=', '=' per row
    b          : list[float]   RHS values (must be made >= 0 by caller)
    var_names  : list[str]     names of the original decision variables
    """

    def __init__(self, c, A, relations, b, var_names=None):
        self.m = len(A)
        self.n = len(A[0])
        self.orig_n = self.n
        self.var_names = var_names or [f"x{i+1}" for i in range(self.n)]
        self.relations = relations

        # Work on rational copies for exactness
        self.A = [[Fraction(v) for v in row] for row in A]
        self.b = [Fraction(v) for v in b]
        self.c_orig = [Fraction(v) for v in c]

        self.col_names = list(self.var_names)
        self.cost = [MVal(0, ci) for ci in self.c_orig]  # cost of each column so far
        self.artificial_cols = []
        self.basis = [None] * self.m

        self._augment_with_slack_surplus_artificial()

    # -------------------------------------------------------------
    def _augment_with_slack_surplus_artificial(self):
        slack_count, surplus_count, art_count = 0, 0, 0

        for i, rel in enumerate(self.relations):
            if rel == "<=":
                slack_count += 1
                col = [Fraction(0)] * self.m
                col[i] = Fraction(1)
                self._add_column(col, MVal(0, 0), f"s{slack_count}")
                self.basis[i] = len(self.col_names) - 1

            elif rel == ">=":
                surplus_count += 1
                col = [Fraction(0)] * self.m
                col[i] = Fraction(-1)
                self._add_column(col, MVal(0, 0), f"su{surplus_count}")

                art_count += 1
                acol = [Fraction(0)] * self.m
                acol[i] = Fraction(1)
                self._add_column(acol, MVal(1, 0), f"a{art_count}")
                self.artificial_cols.append(len(self.col_names) - 1)
                self.basis[i] = len(self.col_names) - 1

            elif rel == "=":
                art_count += 1
                acol = [Fraction(0)] * self.m
                acol[i] = Fraction(1)
                self._add_column(acol, MVal(1, 0), f"a{art_count}")
                self.artificial_cols.append(len(self.col_names) - 1)
                self.basis[i] = len(self.col_names) - 1
            else:
                raise ValueError(f"Unknown relation {rel}")

    def _add_column(self, col, cost_mval, name):
        for i in range(self.m):
            self.A[i].append(col[i])
        self.cost.append(cost_mval)
        self.col_names.append(name)
        self.n += 1

    # -------------------------------------------------------------
    def _zj_minus_cj_row(self):
        """Return list of (Zj - Cj) as MVal for every column (used to pick entering var
        for MINIMISATION: we look for Cj - Zj < 0, i.e. Zj - Cj > 0 meaning improvement)."""
        row = []
        for j in range(self.n):
            zj = MVal(0, 0)
            for i in range(self.m):
                zj = zj + self.cost[self.basis[i]] * self.A[i][j]
            row.append(zj - self.cost[j])   # Zj - Cj
        return row

    def _print_tableau(self, iteration):
        print(f"\n--- Iteration {iteration} ---")
        header = ["Basis", "CB"] + self.col_names + ["RHS"]
        print(("{:>8}" * len(header)).format(*header))
        for i in range(self.m):
            row_name = self.col_names[self.basis[i]]
            cb = self.cost[self.basis[i]]
            vals = [f"{float(v):.2f}" for v in self.A[i]]
            print(("{:>8}" * (2 + self.n + 1)).format(
                row_name, str(cb), *vals, f"{float(self.b[i]):.2f}"))

        zc = self._zj_minus_cj_row()
        print(("{:>8}" * 2).format("Zj-Cj", ""), end="")
        for v in zc:
            print(f"{str(v):>8}", end="")
        print()

    # -------------------------------------------------------------
    def solve(self, verbose=True):
        iteration = 0
        if verbose:
            self._print_tableau(iteration)

        while True:
            zc = self._zj_minus_cj_row()
            # Entering variable: most positive (Zj-Cj) i.e. Cj-Zj most negative
            entering = None
            best = MVal(0, 0)
            for j, val in enumerate(zc):
                if val.m > 0 or (val.m == 0 and val.k > 1e-9):
                    if entering is None or best < val:
                        best = val
                        entering = j

            if entering is None:
                break  # optimal reached

            # Ratio test
            leaving = None
            best_ratio = None
            for i in range(self.m):
                if self.A[i][entering] > 0:
                    ratio = self.b[i] / self.A[i][entering]
                    if best_ratio is None or ratio < best_ratio:
                        best_ratio = ratio
                        leaving = i

            if leaving is None:
                raise RuntimeError("Problem is unbounded.")

            # Pivot
            pivot = self.A[leaving][entering]
            self.A[leaving] = [v / pivot for v in self.A[leaving]]
            self.b[leaving] = self.b[leaving] / pivot

            for i in range(self.m):
                if i != leaving and self.A[i][entering] != 0:
                    factor = self.A[i][entering]
                    self.A[i] = [self.A[i][k] - factor * self.A[leaving][k]
                                 for k in range(self.n)]
                    self.b[i] = self.b[i] - factor * self.b[leaving]

            self.basis[leaving] = entering
            iteration += 1
            if verbose:
                self._print_tableau(iteration)

        return self._extract_solution()

    # -------------------------------------------------------------
    def _extract_solution(self):
        values = [Fraction(0)] * self.n
        for i in range(self.m):
            values[self.basis[i]] = self.b[i]

        # Feasibility check: artificial variables must be zero
        for a in self.artificial_cols:
            if values[a] != 0:
                return {"status": "INFEASIBLE", "values": None, "objective": None}

        x_values = {self.col_names[j]: values[j] for j in range(self.orig_n)}
        obj = sum(self.c_orig[j] * values[j] for j in range(self.orig_n))
        return {"status": "OPTIMAL", "values": x_values, "objective": obj}


# ---------------------------------------------------------------------
#  Case study driver
# ---------------------------------------------------------------------
def main():
    print("=" * 70)
    print("BIG-M SIMPLEX METHOD -- Worked Case Study")
    print("=" * 70)
    print("""
Problem (Minimize):
    Z = 4 x1 + x2

Subject to:
    3 x1 +   x2  =  3
    4 x1 + 3 x2 >=  6
      x1 + 2 x2 <=  4
    x1, x2 >= 0
""")

    c = [4, 1]
    A = [
        [3, 1],
        [4, 3],
        [1, 2],
    ]
    relations = ["=", ">=", "<="]
    b = [3, 6, 4]

    solver = BigMSimplex(c, A, relations, b, var_names=["x1", "x2"])
    result = solver.solve(verbose=True)

    print("\n" + "=" * 70)
    print("FINAL RESULT")
    print("=" * 70)
    if result["status"] == "OPTIMAL":
        for name, val in result["values"].items():
            print(f"  {name} = {float(val):.4f}")
        # any remaining variables (slack/surplus) are not printed;
        # only the decision variables asked for in the case study.
        print(f"\n  Optimal objective value  Z* = {float(result['objective']):.4f}")
    else:
        print("  The problem has NO feasible solution.")

    print("\n(Verified analytically: from 3x1+x2=3 => x2=3-3x1;")
    print(" substituting into the other two constraints gives 0.4 <= x1 <= 0.6,")
    print(" and Z = 4x1+x2 = x1+3, minimised at x1=0.4  =>  x1=0.4, x2=1.8, Z*=3.4)")


if __name__ == "__main__":
    main()
