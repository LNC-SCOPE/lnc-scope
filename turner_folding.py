# -*- coding: utf-8 -*-

# ================================================================
# turner_folding.py
# RNA secondary structure prediction using Turner 2004 nearest-
# neighbor thermodynamic parameters (Mathews et al., 2004).
#
# Replaces the custom pair_energy / stacking_bonus approach in
# lnc-scope.py with experimentally derived parameters.
#
# Reference:
#   Mathews DH, Disney MD, Childs JL, Schroeder SJ, Zuker M,
#   Turner DH. (2004). "Incorporating chemical modification
#   constraints into a dynamic programming algorithm for
#   prediction of RNA secondary structure."
#   PNAS 101(19): 7287-7292.
#
#   Turner DH, Mathews DH. (2010). "NNDB: The nearest neighbor
#   parameter database for predicting stability of nucleic acid
#   secondary structure." Nucleic Acids Research 38: D280-D282.
#
# Units: kcal/mol at 37°C
# ================================================================

import time

# ----------------------------------------------------------------
# PART 1 — TURNER NEAREST-NEIGHBOR STACKING PARAMETERS
#
# Format: STACK[(closing_pair, opening_pair)] = kcal/mol
#
# Read as: the energy contribution when pair (i,j) closes a
# helix and pair (i+1, j-1) is inside it.
#
# Convention: pairs written 5'→3' on the top strand.
# Example: ("AU","GC") means  5'...A-G...3'
#                              3'...U-C...5'
#
# Source: Turner 2004 Table 2 / NNDB (rna.urmc.rochester.edu)
# ----------------------------------------------------------------

STACK = {
    # ---- Watson-Crick / Watson-Crick stacks ----
    # Rows: closing pair (top strand 5'→3')
    # Cols: inner pair  (top strand 5'→3')

    ("AA", "UU"): -0.93,
    ("AC", "UG"): -2.24,
    ("AG", "UC"): -2.08,
    ("AU", "UA"): -1.10,
    ("AU", "UG"): -1.36,   # AU closed, GU inside
    ("AU", "GC"): -2.11,
    ("AU", "CG"): -2.11,
    ("AU", "AU"): -0.93,

    ("CA", "GU"): -2.11,
    ("CC", "GG"): -3.26,
    ("CG", "GA"): -2.35,
    ("CG", "GC"): -3.42,
    ("CG", "GU"): -2.51,
    ("CG", "AU"): -2.11,
    ("CG", "UA"): -2.11,
    ("CU", "GG"): -1.41,   # internal mismatch context — approximate

    ("GA", "CU"): -2.08,
    ("GC", "CG"): -3.26,
    ("GC", "CA"): -2.24,
    ("GC", "GA"): -1.41,   # approximate
    ("GC", "GU"): -1.53,
    ("GC", "AU"): -2.36,
    ("GC", "UA"): -2.36,

    ("GG", "CC"): -3.26,
    ("GU", "CA"): -1.36,
    ("GU", "CG"): -1.53,
    ("GU", "UA"): -1.27,
    ("GU", "AU"): -1.36,
    ("GU", "GU"): -0.50,   # G-U / G-U wobble stack
    ("GU", "UG"): +0.30,   # destabilising
    ("GU", "GC"): -1.53,

    ("UA", "AU"): -1.10,
    ("UA", "AG"): -1.33,
    ("UA", "GA"): -1.36,
    ("UA", "GU"): -1.27,
    ("UA", "GC"): -2.11,
    ("UA", "CG"): -2.11,

    ("UC", "AG"): -2.08,
    ("UG", "AC"): -2.24,
    ("UG", "GC"): -1.53,
    ("UG", "GA"): -2.51,
    ("UG", "UA"): -1.36,
    ("UG", "AU"): -1.36,
    ("UG", "GU"): +0.30,   # destabilising

    ("UU", "AA"): -0.93,
}

# Fallback for any pair combination not explicitly listed.
# Using the average WC stack energy as a conservative estimate.
STACK_DEFAULT = -1.5


def get_stack_energy(seq, i, j):
    """
    Return the stacking energy for the base pair (i,j) stacked
    on the base pair (i+1, j-1).

    Called when both (i,j) and (i+1,j-1) are valid pairs and
    i+1 < j-1.

    Args:
        seq: RNA sequence string (uppercase, U not T)
        i, j: indices of the outer (closing) base pair

    Returns:
        float: stacking energy in kcal/mol
    """
    outer = seq[i] + seq[j]       # e.g. "GC"
    inner = seq[i+1] + seq[j-1]   # e.g. "AU"
    return STACK.get((outer, inner), STACK_DEFAULT)


# ----------------------------------------------------------------
# PART 2 — LOOP PENALTIES (Turner 2004)
#
# Hairpin loops, bulges, and internal loops each have size-
# dependent initiation penalties from experimental data.
# ----------------------------------------------------------------

# Hairpin loop initiation (kcal/mol) indexed by loop size.
# Sizes 0-2 are forbidden (minimum hairpin = 3 nt unpaired).
HAIRPIN_INIT = {
    3:  5.4,
    4:  4.7,
    5:  4.4,
    6:  4.3,
    7:  4.1,
    8:  4.0,
    9:  3.9,
    10: 3.8,
}
HAIRPIN_INIT_DEFAULT = 3.8  # for loops > 10: ~1.75*ln(size/10) + 3.8 approx

# Bulge loop initiation (kcal/mol) indexed by bulge size.
BULGE_INIT = {
    1:  3.8,
    2:  2.8,
    3:  3.2,
    4:  3.6,
    5:  4.0,
    6:  4.4,
}
BULGE_INIT_DEFAULT = 4.4

# Internal loop initiation (kcal/mol) indexed by total loop size.
INTERNAL_INIT = {
    2:  0.0,   # 1x1 internal — handled by mismatch tables; approx here
    4:  0.4,
    6:  0.8,
    8:  1.3,
   10:  1.7,
}
INTERNAL_INIT_DEFAULT = 1.7


def hairpin_energy(size):
    """Initiation energy for a hairpin of `size` unpaired nucleotides."""
    if size < 3:
        return 999.0   # forbidden
    if size <= 10:
        return HAIRPIN_INIT.get(size, HAIRPIN_INIT_DEFAULT)
    # Asymptotic formula for large loops
    return HAIRPIN_INIT_DEFAULT + 1.75 * (37 / 273.15 + 1) * (size / 10.0)


def internal_loop_energy(size1, size2):
    """
    Approximate initiation energy for an internal loop.
    size1 + size2 = total unpaired nucleotides on both sides.
    """
    total = size1 + size2
    if total == 0:
        return 0.0
    return INTERNAL_INIT.get(total, INTERNAL_INIT_DEFAULT)


# ----------------------------------------------------------------
# PART 3 — VALIDITY CHECKS
# ----------------------------------------------------------------

VALID_PAIRS = {
    ("A", "U"), ("U", "A"),
    ("G", "C"), ("C", "G"),
    ("G", "U"), ("U", "G"),   # wobble pair
}


def can_pair(a, b):
    """Return True if nucleotides a and b can form a base pair."""
    return (a, b) in VALID_PAIRS


# ----------------------------------------------------------------
# PART 4 — ZUKER-STYLE DP WITH TURNER PARAMETERS
#
# We implement the standard four-array DP:
#   W[i][j]  = min energy of the best structure on subsequence i..j
#   V[i][j]  = min energy of structures where (i,j) is base-paired
#
# This is a simplified but thermodynamically grounded version;
# it handles hairpin loops, stacked pairs, and bifurcations.
# Internal loop and bulge treatment is approximate (symmetric
# penalty only) — sufficient for accessibility estimation.
# ----------------------------------------------------------------

MIN_HAIRPIN = 3   # minimum unpaired nucleotides in a hairpin

INF = 1e9         # large number representing impossible/unfolded


def fold_turner(seq):
    """
    Fold an RNA sequence using Turner 2004 nearest-neighbor parameters.

    Args:
        seq: str — RNA sequence, uppercase, U not T

    Returns:
        structure: str — dot-bracket notation
        mfe:       float — minimum free energy in kcal/mol
    """
    n = len(seq)

    # V[i][j]: energy of best structure where i and j are paired
    # W[i][j]: energy of best structure on subsequence i..j
    V = [[INF] * n for _ in range(n)]
    W = [[0.0]  * n for _ in range(n)]

    # Fill in order of increasing subsequence length
    for length in range(1, n):
        for i in range(n - length):
            j = i + length

            # --- Compute V[i][j]: i and j must be paired ---
            if can_pair(seq[i], seq[j]) and (j - i - 1) >= MIN_HAIRPIN:

                # Case 1: hairpin loop
                hp = hairpin_energy(j - i - 1)
                V[i][j] = hp

                # Case 2: stacked pair — (i,j) stacked on (i+1,j-1)
                if (j - i - 1) > MIN_HAIRPIN and can_pair(seq[i+1], seq[j-1]):
                    stack_e = get_stack_energy(seq, i, j)
                    interior = V[i+1][j-1]
                    if interior < INF:
                        V[i][j] = min(V[i][j], stack_e + interior)

                # Case 3: internal loop / bulge (simplified: skip 1-2 nt each side)
                for skip_i in range(1, 4):
                    for skip_j in range(1, 4):
                        ni, nj = i + skip_i, j - skip_j
                        if ni >= nj or not can_pair(seq[ni], seq[nj]):
                            continue
                        if (nj - ni - 1) < MIN_HAIRPIN:
                            continue
                        il_e = internal_loop_energy(skip_i - 1, skip_j - 1)
                        if V[ni][nj] < INF:
                            V[i][j] = min(V[i][j], il_e + V[ni][nj])

                # Case 4: multi-loop / bifurcation — (i,j) encloses two substructures
                for k in range(i + 1, j):
                    if W[i+1][k] < INF and W[k+1][j-1] < INF:
                        V[i][j] = min(V[i][j], W[i+1][k] + W[k+1][j-1])

            # --- Compute W[i][j] ---

            # Option 1: i or j left unpaired
            best_w = min(
                W[i+1][j] if i+1 <= j else 0.0,
                W[i][j-1] if i <= j-1 else 0.0
            )

            # Option 2: i and j paired (close a helix here)
            if V[i][j] < INF:
                best_w = min(best_w, V[i][j])

            # Option 3: bifurcation within W
            for k in range(i, j):
                lw = W[i][k]   if i <= k   else 0.0
                rw = W[k+1][j] if k+1 <= j else 0.0
                if lw < INF and rw < INF:
                    best_w = min(best_w, lw + rw)

            W[i][j] = best_w

    # --- Traceback ---
    structure = ['.'] * n

    def traceback_V(i, j):
        """Traceback through V — i and j are paired."""
        if i >= j or not can_pair(seq[i], seq[j]):
            return
        structure[i] = '('
        structure[j] = ')'

        hp = hairpin_energy(j - i - 1)
        if abs(V[i][j] - hp) < 1e-6:
            return  # hairpin: nothing inside

        # Stacked pair?
        if (j - i - 1) > MIN_HAIRPIN and can_pair(seq[i+1], seq[j-1]):
            se = get_stack_energy(seq, i, j)
            if V[i+1][j-1] < INF and abs(V[i][j] - se - V[i+1][j-1]) < 1e-6:
                traceback_V(i+1, j-1)
                return

        # Internal loop / bulge?
        for skip_i in range(1, 4):
            for skip_j in range(1, 4):
                ni, nj = i + skip_i, j - skip_j
                if ni >= nj or not can_pair(seq[ni], seq[nj]):
                    continue
                il_e = internal_loop_energy(skip_i - 1, skip_j - 1)
                if V[ni][nj] < INF and abs(V[i][j] - il_e - V[ni][nj]) < 1e-6:
                    traceback_V(ni, nj)
                    return

        # Multi-loop bifurcation?
        for k in range(i+1, j):
            if W[i+1][k] < INF and W[k+1][j-1] < INF:
                if abs(V[i][j] - W[i+1][k] - W[k+1][j-1]) < 1e-6:
                    traceback_W(i+1, k)
                    traceback_W(k+1, j-1)
                    return

    def traceback_W(i, j):
        """Traceback through W — best structure on i..j."""
        if i >= j:
            return

        # Unpaired?
        if i+1 <= j and abs(W[i][j] - W[i+1][j]) < 1e-6:
            traceback_W(i+1, j)
            return
        if i <= j-1 and abs(W[i][j] - W[i][j-1]) < 1e-6:
            traceback_W(i, j-1)
            return

        # Paired at (i,j)?
        if V[i][j] < INF and abs(W[i][j] - V[i][j]) < 1e-6:
            traceback_V(i, j)
            return

        # Bifurcation?
        for k in range(i, j):
            lw = W[i][k]   if i <= k   else 0.0
            rw = W[k+1][j] if k+1 <= j else 0.0
            if lw < INF and rw < INF and abs(W[i][j] - lw - rw) < 1e-6:
                traceback_W(i, k)
                traceback_W(k+1, j)
                return

    if n > 0:
        traceback_W(0, n-1)

    mfe = W[0][n-1] if n > 1 else 0.0
    return ''.join(structure), mfe
