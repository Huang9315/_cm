import numpy as np
from scipy.linalg import lu

# 設定隨機種子以便重現結果
np.random.seed(42)

# ==========================================
# 1. 遞迴計算行列式 (Recursive Determinant)
# ==========================================
def recursive_det(matrix):
    """
    使用拉普拉斯展開 (Laplace Expansion) 遞迴計算行列式。
    注意：時間複雜度為 O(n!)，僅適合教學或極小矩陣。
    """
    A = np.array(matrix)
    n = A.shape[0]

    # Base Case: 2x2 矩陣
    if n == 2:
        return A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    
    det = 0
    # 沿著第一列 (row 0) 展開
    for col in range(n):
        # 建立子矩陣 (Minor)：移除第 0 列和目前的 col 行
        sub_matrix = np.delete(np.delete(A, 0, axis=0), col, axis=1)
        
        # 公式: (-1)^(i+j) * a_ij * det(sub_matrix)
        sign = (-1) ** col
        det += sign * A[0, col] * recursive_det(sub_matrix)
        
    return det

# ==========================================
# 2. LU 分解計算行列式
# ==========================================
def det_via_lu(A):
    """
    透過 LU 分解計算行列式。
    Det(A) = Det(P) * Det(L) * Det(U)
    """
    # 使用 SciPy 進行 LU 分解: A = P @ L @ U
    P, L, U = lu(A)
    
    # 1. U 是上三角，行列式 = 對角線乘積
    det_U = np.prod(np.diag(U))
    
    # 2. L 是下三角且對角線為 1 (SciPy 特性)，行列式 = 1
    det_L = 1
    
    # 3. P 是置換矩陣，行列式為 1 或 -1
    det_P = np.linalg.det(P) 
    
    return det_P * det_L * det_U

# ==========================================
# 3. 透過特徵值分解實作 SVD
# ==========================================
def svd_via_eig(A):
    """
    利用 A^T A 的特徵值分解來計算 SVD (A = U Σ V^T)。
    """
    # 1. 計算 A^T A
    ATA = A.T @ A
    
    # 2. 對 A^T A 做特徵值分解 (eigh 適用於對稱矩陣)
    evals, evecs = np.linalg.eigh(ATA)
    
    # 3. 排序 (由大到小)
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    V = evecs[:, idx] # 這是 V，不是 Vt
    
    # 4. 計算奇異值 (特徵值的平方根)
    singular_values = np.sqrt(np.abs(evals))
    
    # 為了計算 U，過濾掉接近 0 的奇異值
    tolerance = 1e-10
    nonzero_indices = singular_values > tolerance
    sigma_nonzero = singular_values[nonzero_indices]
    V_nonzero = V[:, nonzero_indices]
    
    # 5. 計算 U: u_i = (A v_i) / sigma_i
    # 注意：這裡只計算出對應非零奇異值的 U (Compact SVD)
    U = (A @ V_nonzero) / sigma_nonzero
    
    return U, sigma_nonzero, V.T

# ==========================================
# 4. 主成份分析 (PCA)
# ==========================================
def simple_pca(X, n_components=2):
    """
    執行 PCA 降維。
    回傳: 降維後的數據, 特徵值, 特徵向量
    """
    # 1. 數據中心化
    mean_vec = np.mean(X, axis=0)
    X_centered = X - mean_vec
    
    # 2. 計算共變異數矩陣
    cov_matrix = np.cov(X_centered, rowvar=False)
    
    # 3. 特徵值分解
    eigen_vals, eigen_vecs = np.linalg.eigh(cov_matrix)
    
    # 4. 排序
    sorted_index = np.argsort(eigen_vals)[::-1]
    sorted_eigenvals = eigen_vals[sorted_index]
    sorted_eigenvecs = eigen_vecs[:, sorted_index]
    
    # 5. 選取前 k 個主成份
    eigenvector_subset = sorted_eigenvecs[:, 0:n_components]
    
    # 6. 投影
    X_reduced = np.dot(X_centered, eigenvector_subset)
    
    return X_reduced, sorted_eigenvals, eigenvector_subset


# ==========================================
# 主程式 (Main Execution / Testing)
# ==========================================
if __name__ == "__main__":
    print("=== 線性代數演算法實作展示 ===\n")

    # --- 測試 1: 行列式計算 (遞迴 vs LU vs Numpy) ---
    print("[1] 行列式計算比較")
    matrix_size = 5 # 建議不要超過 10，否則遞迴會跑很久
    A_det = np.random.randint(1, 10, (matrix_size, matrix_size))
    # 為了演示遞迴，我們先用一個極小的 3x3
    A_small = np.array([[6, 1, 1], [4, -2, 5], [2, 8, 7]])
    
    print(f"測試矩陣 (3x3):\n{A_small}")
    print(f"-> 遞迴法結果: {recursive_det(A_small)}")
    print(f"-> LU 分解法結果: {det_via_lu(A_small):.4f}")
    print(f"-> Numpy 標準答案: {np.linalg.det(A_small):.4f}")
    print("-" * 30)

    # --- 測試 2: 分解還原驗證 ---
    print("\n[2] 矩陣分解與還原驗證")
    A_verify = np.random.rand(4, 4)
    
    # A. LU 分解驗證
    P, L, U = lu(A_verify)
    A_lu_rec = P @ L @ U
    print(f"LU 還原成功? {np.allclose(A_verify, A_lu_rec)}")
    
    # B. 特徵值分解驗證
    vals, vecs = np.linalg.eig(A_verify)
    Q = vecs
    Lambda = np.diag(vals)
    Q_inv = np.linalg.inv(Q)
    A_eig_rec = Q @ Lambda @ Q_inv
    print(f"Eigen 還原成功? {np.allclose(A_verify, A_eig_rec)}")
    
    # C. SVD 分解驗證
    U_s, S_s, Vt_s = np.linalg.svd(A_verify)
    Sigma_s = np.diag(S_s)
    A_svd_rec = U_s @ Sigma_s @ Vt_s
    print(f"SVD 還原成功? {np.allclose(A_verify, A_svd_rec)}")
    print("-" * 30)

    # --- 測試 3: 手刻 SVD (透過 Eigen) ---
    print("\n[3] 透過特徵值分解實作 SVD")
    A_rect = np.array([[3.0, 2.0, 2.0], [2.0, 3.0, -2.0]])
    print(f"原矩陣 (2x3):\n{A_rect}")
    
    my_U, my_S, my_Vt = svd_via_eig(A_rect)
    np_U, np_S, np_Vt = np.linalg.svd(A_rect)
    
    print(f"-> 手刻 SVD 奇異值: {my_S}")
    print(f"-> Numpy SVD 奇異值: {np_S}")
    print(f"-> 數值是否一致? {np.allclose(my_S, np_S)}")
    print("-" * 30)

    # --- 測試 4: PCA 主成份分析 ---
    print("\n[4] PCA 主成份分析")
    # 產生高度相關的 2D 數據
    x1 = np.random.normal(0, 1, 100)
    x2 = x1 * 0.8 + np.random.normal(0, 0.4, 100)
    X_pca = np.vstack((x1, x2)).T
    
    X_reduced, evals, evecs = simple_pca(X_pca, n_components=1)
    
    print(f"原始數據形狀: {X_pca.shape}")
    print(f"降維後形狀: {X_reduced.shape}")
    print(f"第一主成份解釋變異量: {evals[0] / sum(evals):.2%}")
    print("=== 展示結束 ===")