"""
TRANSPORTATION PROBLEM -- VAM (Vogel's Approximation Method) for the
initial basic feasible solution, followed by the MODI (Modified
Distribution) Method to test optimality and iteratively improve the
allocation until the minimum-cost shipment plan is found.

Case study (a classic, widely used textbook transportation problem):

    Three factories (sources) supply four warehouses (destinations).

                 W1    W2    W3    W4   | Supply
        F1        4     6     8     6   |   50
        F2        3     5     2     5   |   60
        F3        3     9     6     5   |   25
        --------------------------------------------
        Demand   30    40    50    15   |  135 / 135   (balanced)

Goal: find the shipment plan (how many units to ship from each
factory to each warehouse) that minimises total transportation cost.

The program is written generically (works for any supply/demand/cost
table, and automatically balances an unbalanced problem by adding a
dummy source/destination with zero cost) and prints every step:
  1. VAM initial allocation (with penalty calculations)
  2. MODI optimality test (u_i, v_j, and opportunity costs) and the
     stepping-stone improvement loop, iteration by iteration
  3. Final optimal shipment plan and minimum total cost
"""

import copy
from typing import List


INF = float("inf")


# ---------------------------------------------------------------------
# Utility printing
# ---------------------------------------------------------------------
def print_table(cost, supply, demand, alloc=None, title=""):
    m, n = len(cost), len(cost[0])
    if title:
        print(f"\n{title}")
    header = "        " + "".join(f"{'D'+str(j+1):>8}" for j in range(n)) + "   Supply"
    print(header)
    for i in range(m):
        row = f"S{i+1:>3} :  "
        for j in range(n):
            cell = f"{cost[i][j]:>3}"
            if alloc is not None and alloc[i][j] > 0:
                cell += f"({alloc[i][j]})"
            row += f"{cell:>8}"
        row += f"{supply[i]:>9}"
        print(row)
    demand_row = "Demand:  " + "".join(f"{d:>8}" for d in demand)
    print(demand_row)


# ---------------------------------------------------------------------
# Step 0 : balance the problem if total supply != total demand
# ---------------------------------------------------------------------
def balance_problem(cost, supply, demand):
    cost = copy.deepcopy(cost)
    supply = list(supply)
    demand = list(demand)
    total_s, total_d = sum(supply), sum(demand)

    if total_s == total_d:
        return cost, supply, demand

    if total_s < total_d:
        print(f"\n[Balancing] Supply ({total_s}) < Demand ({total_d}); "
              f"adding a dummy source with 0 cost and supply {total_d - total_s}.")
        cost.append([0] * len(demand))
        supply.append(total_d - total_s)
    else:
        print(f"\n[Balancing] Supply ({total_s}) > Demand ({total_d}); "
              f"adding a dummy destination with 0 cost and demand {total_s - total_d}.")
        for row in cost:
            row.append(0)
        demand.append(total_s - total_d)

    return cost, supply, demand


# ---------------------------------------------------------------------
# STEP 1 : Vogel's Approximation Method (VAM) - initial BFS
# ---------------------------------------------------------------------
def vam_initial_bfs(cost, supply, demand):
    m, n = len(cost), len(cost[0])
    supply = list(supply)
    demand = list(demand)
    alloc = [[0] * n for _ in range(m)]
    row_done = [False] * m
    col_done = [False] * n

    print("\n" + "=" * 70)
    print("STEP 1: Vogel's Approximation Method (VAM) -- Initial BFS")
    print("=" * 70)

    step = 1
    remaining_rows = m
    remaining_cols = n

    while remaining_rows > 0 and remaining_cols > 0:
        # If only one row or one column remains, allocate directly (no
        # penalty computation is meaningful any more).
        if remaining_rows == 1 or remaining_cols == 1:
            for i in range(m):
                if row_done[i]:
                    continue
                for j in range(n):
                    if col_done[j]:
                        continue
                    qty = min(supply[i], demand[j])
                    if qty > 0:
                        alloc[i][j] += qty
                        supply[i] -= qty
                        demand[j] -= qty
                    if supply[i] == 0:
                        row_done[i] = True
                        remaining_rows -= 1
                    if demand[j] == 0:
                        col_done[j] = True
                        remaining_cols -= 1
            break

        # ---- Row penalties ----
        row_penalty = []
        for i in range(m):
            if row_done[i]:
                row_penalty.append(-1)
                continue
            vals = sorted(cost[i][j] for j in range(n) if not col_done[j])
            pen = (vals[1] - vals[0]) if len(vals) >= 2 else vals[0]
            row_penalty.append(pen)

        # ---- Column penalties ----
        col_penalty = []
        for j in range(n):
            if col_done[j]:
                col_penalty.append(-1)
                continue
            vals = sorted(cost[i][j] for i in range(m) if not row_done[i])
            pen = (vals[1] - vals[0]) if len(vals) >= 2 else vals[0]
            col_penalty.append(pen)

        print(f"\n-- VAM Step {step} --")
        print("Row penalties   :", {f"S{i+1}": row_penalty[i] for i in range(m) if not row_done[i]})
        print("Column penalties:", {f"D{j+1}": col_penalty[j] for j in range(n) if not col_done[j]})

        max_row_pen = max((p for i, p in enumerate(row_penalty) if not row_done[i]), default=-1)
        max_col_pen = max((p for j, p in enumerate(col_penalty) if not col_done[j]), default=-1)

        if max_row_pen >= max_col_pen:
            # pick the row with max penalty
            sel_i = next(i for i in range(m) if not row_done[i] and row_penalty[i] == max_row_pen)
            sel_j = min((j for j in range(n) if not col_done[j]), key=lambda j: cost[sel_i][j])
        else:
            sel_j = next(j for j in range(n) if not col_done[j] and col_penalty[j] == max_col_pen)
            sel_i = min((i for i in range(m) if not row_done[i]), key=lambda i: cost[i][sel_j])

        qty = min(supply[sel_i], demand[sel_j])
        alloc[sel_i][sel_j] += qty
        supply[sel_i] -= qty
        demand[sel_j] -= qty
        print(f"Selected cell (S{sel_i+1}, D{sel_j+1}) with lowest cost {cost[sel_i][sel_j]} "
              f"-> allocate min(supply, demand) = {qty} units")

        if supply[sel_i] == 0 and not row_done[sel_i]:
            row_done[sel_i] = True
            remaining_rows -= 1
        if demand[sel_j] == 0 and not col_done[sel_j]:
            col_done[sel_j] = True
            remaining_cols -= 1

        step += 1

    print("\nInitial BFS obtained via VAM:")
    print_table(cost, [sum(alloc[i]) for i in range(m)],
                [sum(alloc[i][j] for i in range(m)) for j in range(n)], alloc,
                title=None)
    total_cost = sum(alloc[i][j] * cost[i][j] for i in range(m) for j in range(n))
    print(f"\nInitial VAM transportation cost = {total_cost}")
    return alloc


# ---------------------------------------------------------------------
# STEP 2 : MODI method - optimality test & improvement
# ---------------------------------------------------------------------
def find_basic_cells(alloc):
    m, n = len(alloc), len(alloc[0])
    cells = [(i, j) for i in range(m) for j in range(n) if alloc[i][j] > 0]
    return cells


def ensure_non_degenerate(alloc, cost):
    """A transportation BFS must have exactly m+n-1 basic (allocated)
    cells. If fewer (degenerate case), add an epsilon (very small)
    allocation at the lowest-cost unoccupied cell that does not form a
    loop, so MODI's u_i/v_j system can be solved."""
    m, n = len(alloc), len(alloc[0])
    needed = m + n - 1
    basic = find_basic_cells(alloc)
    EPS = 1e-6
    while len(basic) < needed:
        candidates = sorted(
            ((i, j) for i in range(m) for j in range(n) if alloc[i][j] == 0),
            key=lambda ij: cost[ij[0]][ij[1]]
        )
        if not candidates:
            break
        i, j = candidates[0]
        alloc[i][j] = EPS
        basic = find_basic_cells(alloc)
    return alloc


def compute_u_v(cost, alloc):
    m, n = len(cost), len(cost[0])
    u = [None] * m
    v = [None] * n
    u[0] = 0
    basic = find_basic_cells(alloc)

    changed = True
    while changed:
        changed = False
        for (i, j) in basic:
            if u[i] is not None and v[j] is None:
                v[j] = cost[i][j] - u[i]
                changed = True
            elif v[j] is not None and u[i] is None:
                u[i] = cost[i][j] - v[j]
                changed = True
    # Any still-unlinked rows/cols (can happen in degenerate/disconnected
    # cases) default to 0 to keep the algorithm moving.
    u = [x if x is not None else 0 for x in u]
    v = [x if x is not None else 0 for x in v]
    return u, v


def find_loop(alloc, start):
    """Find a closed loop for stepping-stone / MODI starting at the
    (currently unoccupied) 'start' cell, alternating horizontal and
    vertical moves through basic (allocated) cells."""
    m, n = len(alloc), len(alloc[0])
    basic_cells = set(find_basic_cells(alloc))
    basic_cells.add(start)

    def get_row_cells(r, exclude):
        return [c for c in basic_cells if c[0] == r and c != exclude]

    def get_col_cells(c_, exclude):
        return [c for c in basic_cells if c[1] == c_ and c != exclude]

    path = [start]

    def dfs(current, come_from_row, visited):
        if len(path) >= 4 and current == start:
            return True
        neighbours = (get_col_cells(current[1], current) if come_from_row
                      else get_row_cells(current[0], current))
        for nxt in neighbours:
            if nxt == start and len(path) >= 3:
                path.append(nxt)
                return True
            if nxt in visited:
                continue
            path.append(nxt)
            visited.add(nxt)
            if dfs(nxt, not come_from_row, visited):
                return True
            path.pop()
            visited.discard(nxt)
        return False

    visited = {start}
    if dfs(start, False, visited):
        return path[:-1]
    return None


def modi_optimize(cost, alloc, verbose=True):
    m, n = len(cost), len(cost[0])
    alloc = [row[:] for row in alloc]
    alloc = ensure_non_degenerate(alloc, cost)

    print("\n" + "=" * 70)
    print("STEP 2: MODI (Modified Distribution) Method -- Optimality Test")
    print("=" * 70)

    iteration = 0
    while True:
        iteration += 1
        u, v = compute_u_v(cost, alloc)
        print(f"\n-- MODI Iteration {iteration} --")
        print("u_i :", {f"S{i+1}": u[i] for i in range(m)})
        print("v_j :", {f"D{j+1}": v[j] for j in range(n)})

        # Opportunity cost for every non-basic (unallocated) cell:
        #   d_ij = c_ij - (u_i + v_j).   Optimal when all d_ij >= 0.
        opp_cost = [[None] * n for _ in range(m)]
        most_negative = 0
        enter_cell = None
        for i in range(m):
            for j in range(n):
                if alloc[i][j] == 0:
                    d = cost[i][j] - (u[i] + v[j])
                    opp_cost[i][j] = d
                    if d < most_negative:
                        most_negative = d
                        enter_cell = (i, j)

        print("Opportunity costs d_ij = c_ij - (u_i+v_j) for unallocated cells:")
        for i in range(m):
            row_display = []
            for j in range(n):
                if opp_cost[i][j] is not None:
                    row_display.append(f"D{j+1}:{opp_cost[i][j]:+.0f}")
            print(f"  S{i+1}: " + ", ".join(row_display))

        if enter_cell is None:
            print("\nAll opportunity costs d_ij >= 0  =>  current solution is OPTIMAL.")
            break

        print(f"\nMost negative opportunity cost is at cell (S{enter_cell[0]+1}, "
              f"D{enter_cell[1]+1}) = {most_negative:+.0f}  -> this cell enters the basis.")

        loop = find_loop(alloc, enter_cell)
        if loop is None:
            print("Could not find a closed loop (degenerate case) -- stopping.")
            break

        print("Closed loop (stepping-stone path):",
              [f"(S{i+1},D{j+1})" for i, j in loop])

        # Assign +/- alternately starting with '+' at the entering cell
        minus_cells = loop[1::2]
        theta = min(alloc[i][j] for (i, j) in minus_cells)
        print(f"Minimum allocation among '-' cells (theta) = {theta:.4g}  -> "
              f"this many units are reallocated around the loop.")

        for idx, (i, j) in enumerate(loop):
            if idx % 2 == 0:
                alloc[i][j] += theta
            else:
                alloc[i][j] -= theta

        # Clean near-zero / epsilon allocations
        for i in range(m):
            for j in range(n):
                if abs(alloc[i][j]) < 1e-7:
                    alloc[i][j] = 0

        total_cost = sum(alloc[i][j] * cost[i][j] for i in range(m) for j in range(n))
        print(f"Updated total cost after this iteration: {total_cost:.4g}")

    # final clean-up: round tiny epsilon leftovers to 0 for display
    for i in range(m):
        for j in range(n):
            if alloc[i][j] < 1e-4:
                alloc[i][j] = 0
    return alloc


# ---------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------
def main():
    print("=" * 70)
    print("TRANSPORTATION PROBLEM -- VAM (initial BFS) + MODI (optimisation)")
    print("=" * 70)

    cost = [
        [4, 6, 8, 6],
        [3, 5, 2, 5],
        [3, 9, 6, 5],
    ]
    supply = [50, 60, 25]
    demand = [30, 40, 50, 15]

    print("\nCost matrix, supply and demand:")
    print_table(cost, supply, demand)

    cost, supply, demand = balance_problem(cost, supply, demand)

    vam_alloc = vam_initial_bfs(cost, supply, demand)
    optimal_alloc = modi_optimize(cost, vam_alloc)

    m, n = len(cost), len(cost[0])
    print("\n" + "=" * 70)
    print("FINAL OPTIMAL SHIPMENT PLAN")
    print("=" * 70)
    print_table(cost, supply, demand, optimal_alloc)

    total_cost = sum(optimal_alloc[i][j] * cost[i][j] for i in range(m) for j in range(n))
    print("\nShipment plan (source -> destination : units):")
    for i in range(m):
        for j in range(n):
            if optimal_alloc[i][j] > 0:
                src = f"F{i+1}" if i < 3 else f"Dummy-S{i+1}"
                dst = f"W{j+1}" if j < 4 else f"Dummy-D{j+1}"
                print(f"  {src} -> {dst} : {optimal_alloc[i][j]:.0f} units "
                      f"(unit cost {cost[i][j]})")

    print(f"\nMINIMUM TOTAL TRANSPORTATION COST = {total_cost:.0f}")


if __name__ == "__main__":
    main()
