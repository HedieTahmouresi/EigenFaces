"""E7 -- Solver validation: power iteration vs. eigh.

Claim tested: is the from-scratch solve_power correct?
Measures: eigenvalue error, orthogonality error (|u^T u_ref|, never raw
vectors -- eigenvectors are defined up to sign), runtime.
Validate at k=10 first. Eigenvalues matching to ~1e-8 and |u^T u_ref| ~ 1
means correct. Do NOT debug this at k=279.
Output: results rows in results/results.md.
"""

if __name__ == "__main__":
    raise NotImplementedError
