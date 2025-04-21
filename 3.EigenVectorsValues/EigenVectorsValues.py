import numpy as np
import time

# ==============================
# ÖZDEĞER VE ÖZVEKTÖR HESAPLAMA
# ==============================

# -------------------------------
# CUSTOM FUNCTION
# -------------------------------

def minor(matrix, row, col):
    """Verilen matristen bir satır ve sütunu çıkararak minor matris oluşturur."""
    return [
        [elem for j, elem in enumerate(r) if j != col]
        for i, r in enumerate(matrix) if i != row
    ]

def determinant(matrix):
    """Kare matrisin determinantını rekürsif olarak hesaplar."""
    n = len(matrix)
    if n == 1:
        return matrix[0][0]
    if n == 2:
        return matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0]
    det = 0
    for c in range(n):
        det += ((-1) ** c) * matrix[0][c] * determinant(minor(matrix, 0, c))
    return det

def identity(n):
    """n x n boyutlu birim matris oluşturur."""
    return [[1 if i == j else 0 for j in range(n)] for i in range(n)]

def build_characteristic_matrix(A):
    """Karakteristik matris A - λI'yi çok terimli gösterimle oluşturur."""
    n = len(A)
    I = identity(n)
    return [[[A[i][j], -I[i][j]] for j in range(n)] for i in range(n)]

def poly_add(p1, p2):
    """İki polinomu toplar."""
    length = max(len(p1), len(p2))
    p1 += [0] * (length - len(p1))
    p2 += [0] * (length - len(p2))
    return [a + b for a, b in zip(p1, p2)]

def poly_multiply(p1, p2):
    """İki polinomu çarpar."""
    result = [0]*(len(p1)+len(p2)-1)
    for i in range(len(p1)):
        for j in range(len(p2)):
            result[i+j] += p1[i]*p2[j]
    return result

def determinant_polynomial(matrix):
    """Karakteristik polinomu hesaplar."""
    n = len(matrix)
    if n == 2:
        a = matrix[0][0]
        b = matrix[0][1]
        c = matrix[1][0]
        d = matrix[1][1]
        return poly_add(poly_multiply(a, d), [-x for x in poly_multiply(b, c)])
    else:
        result = [0]
        for col in range(n):
            sign = (-1) ** col
            submatrix = minor(matrix, 0, col)
            coeff = matrix[0][col]
            sub_poly = determinant_polynomial(submatrix)
            term = poly_multiply(coeff, sub_poly)
            result = poly_add(result, [sign * t for t in term])
        return result

def characteristic_polynomial(matrix):
    """Karakteristik polinomun katsayılarını döndürür."""
    A_minus_lambdaI = build_characteristic_matrix(matrix)
    return determinant_polynomial(A_minus_lambdaI)

def find_eigenvalues(matrix):
    """Manuel yöntemle özdeğerleri hesaplar (kök bulma)."""
    poly = characteristic_polynomial(matrix)
    poly = [float(c) for c in poly]
    return np.roots(poly[::-1])

def find_eigenvectors(matrix, eigenvalues):
    """Her özdeğer için özvektör hesaplar (Ax = λx çözümü)."""
    eigenvectors = []
    A = np.array(matrix)
    for val in eigenvalues:
        I = np.eye(A.shape[0])
        mat = A - val * I
        u, s, vh = np.linalg.svd(mat)
        null_mask = (s <= 1e-10)
        vecs = vh.T[:, null_mask]
        if vecs.size == 0:
            vecs = np.zeros((A.shape[0], 1))
        eigenvectors.append(vecs[:, 0])
    return np.array(eigenvectors).T

# -------------------------------
# MATRİS ve HESAPLAMALAR
# -------------------------------
A = [[6, 1, -1],
     [0, 7, 0],
     [3, -1, 2]]

A_np = np.array(A)

# NumPy ile hesaplama (yüksek hassasiyetle)
start_np = time.perf_counter()
eigenvalues_np, eigenvectors_np = np.linalg.eig(A_np)
end_np = time.perf_counter()

# Custom fonksiyonlarla hesaplama
start_custom = time.perf_counter()
eigenvalues_custom = find_eigenvalues(A)
eigenvectors_custom = find_eigenvectors(A, eigenvalues_custom)
end_custom = time.perf_counter()

# -------------------------------
# SONUÇLARI GÖSTER
# -------------------------------
print("=== NumPy ===")
print("Özdeğerler:", np.round(eigenvalues_np, 4))
print("Özvektörler:\n", np.round(eigenvectors_np, 4))

print("\n=== Custom ===")
print("Özdeğerler:", np.round(eigenvalues_custom, 4))
print("Özvektörler:\n", np.round(eigenvectors_custom, 4))

print("\n=== Süre Karşılaştırması ===")
print(f"NumPy süresi: {end_np - start_np:.10f} saniye")
print(f"Custom süresi: {end_custom - start_custom:.10f} saniye")
