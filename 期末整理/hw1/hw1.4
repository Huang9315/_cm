import numpy as np

def root(c):
    # Convert to list copy so original isn't modified
    c = list(c)

    # Remove trailing zeros in highest-degree coefficients
    while len(c) > 1 and abs(c[-1]) < 1e-14:
        c.pop()

    n = len(c) - 1  # polynomial degree

    if n == 0:
        return []  # constant polynomial has no roots
    
    if n == 1:
        # ax + b = 0 → -b / a
        return [-c[0] / c[1]]

    # Normalize polynomial so leading term is 1
    c = [ci / c[-1] for ci in c]

    # Build companion matrix
    companion = np.zeros((n, n))
    companion[1:, :-1] = np.eye(n - 1)
    companion[0, :] = -np.array(c[:-1])

    # Eigenvalues = roots
    roots = np.linalg.eigvals(companion)
    return roots

# Example: x^3 - 6x^2 + 11x - 6 has roots 1, 2, 3
coeffs = [-6, 11, -6, 1]
print(root(coeffs))