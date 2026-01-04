import numpy as np
from collections import defaultdict

def solve_ode_general(coefficients):
    """
    求解常係數齊次常微分方程的通解。
    輸入: coefficients (list): 方程係數，從最高階到零階。例如 y'' - 3y' + 2y = 0 -> [1, -3, 2]
    輸出: 通解的字串形式
    """
    
    # 1. 求解特徵方程的根
    # np.roots 接受係數 [a_n, ..., a_0] 對應 a_n * x^n + ... + a_0 = 0
    roots = np.roots(coefficients)
    
    # 2. 對根進行分組與整理 (處理數值誤差與重根)
    # 由於浮點數運算的誤差，我們需要設定一個容許值 (tolerance) 來判斷根是否相同或是否為實數
    tol = 1e-5
    
    processed_roots = []
    
    # 標記已處理過的根索引，避免重複處理
    handled_indices = set()
    
    # 排序根，讓輸出順序穩定 (實部優先，虛部次之)
    # 將複數根轉為 tuple 以便排序 (real, imag)
    roots_sorted = sorted(roots, key=lambda x: (x.real, x.imag))
    
    real_groups = []    # 存放 (root_value, multiplicity)
    complex_groups = [] # 存放 (root_value, multiplicity)，只存共軛對中的一個 (虛部 > 0)

    for i, r in enumerate(roots_sorted):
        if i in handled_indices:
            continue
            
        # 找出所有與 r 相同的根 (考慮 tolerance)
        count = 0
        current_indices = []
        for j, other in enumerate(roots_sorted):
            if j in handled_indices:
                continue
            if abs(r - other) < tol:
                count += 1
                current_indices.append(j)
        
        # 標記這些根已處理
        for idx in current_indices:
            handled_indices.add(idx)
            
        # 判斷是實根還是複數根
        if abs(r.imag) < tol:
            real_groups.append((r.real, count))
        else:
            # 複數根通常成對出現 (a+bi, a-bi)。我們只處理虛部為正的那一個來生成 sin/cos
            # 如果目前這個根的虛部是負的，且我們假設係數為實數，則它應該已經被對應的正虛部根處理過
            # 但為了保險起見（或根排序導致負的先出現），我們這裡做個檢查
            if r.imag > 0:
                complex_groups.append((r, count))
            elif r.imag < -tol:
                # 這是負虛部的根，檢查是否已有對應的正虛部根被加入 (簡單略過，假設成對)
                # 在實係數多項式中，共軛根重數相同。
                pass

    # 3. 根據根的類型生成通解字串
    terms = []
    
    # 處理實根
    for r_val, count in real_groups:
        # 格式化數字，去除不必要的 .0
        r_str = f"{r_val:.5g}"
        
        for k in range(count):
            # 形式: x^k * e^(rx)
            term = ""
            if k == 1:
                term += "x"
            elif k > 1:
                term += f"x^{k}"
            
            # 處理 e 的指數部分
            # 如果 r 為 0，e^0 = 1，省略不寫
            if abs(r_val) > tol:
                term += f"e^({r_str}x)"
            elif k == 0: # 如果 r=0 且 k=0，項為常數 1
                 # 通常我們會寫 C_i，但這裡只生成變數部分，若全空則補 "1" (雖然下面邏輯是 C_i * term)
                 # 但為了美觀，若 term 為空 (即 C * 1)，保持空字串讓 C_i 直接顯示即可? 
                 # 不，如果不加任何變數，看起來像常數項。
                 pass
            
            terms.append(term)

    # 處理複數根 (a +/- bi)
    for r_val, count in complex_groups:
        alpha = r_val.real
        beta = r_val.imag
        
        alpha_str = f"{alpha:.5g}"
        beta_str = f"{beta:.5g}"
        
        for k in range(count):
            # 形式: x^k * e^(ax) * cos(bx) 和 x^k * e^(ax) * sin(bx)
            prefix = ""
            if k == 1:
                prefix += "x"
            elif k > 1:
                prefix += f"x^{k}"
            
            exp_part = ""
            if abs(alpha) > tol:
                exp_part = f"e^({alpha_str}x)"
            
            # 組合 cos 項
            term_cos = f"{prefix}{exp_part}cos({beta_str}x)"
            terms.append(term_cos)
            
            # 組合 sin 項
            term_sin = f"{prefix}{exp_part}sin({beta_str}x)"
            terms.append(term_sin)

    # 4. 加上係數 C_1, C_2... 並組合
    final_parts = []
    for i, term in enumerate(terms):
        c_part = f"C_{i+1}"
        # 如果 term 不為空，中間不加空格直接連起來 (如 C_1e^x) 或視需求調整
        final_parts.append(f"{c_part}{term}")
        
    result = " + ".join(final_parts)
    return f"y(x) = {result}"

# --- 測試程式碼 (與 Issue 內容一致) ---

# 範例測試 (1): 實數單根
print("--- 實數單根範例 ---")
coeffs1 = [1, -3, 2]
print(f"方程係數: {coeffs1}")
print(solve_ode_general(coeffs1))

# 範例測試 (2): 實數重根
print("\n--- 實數重根範例 ---")
coeffs2 = [1, -4, 4]
print(f"方程係數: {coeffs2}")
print(solve_ode_general(coeffs2))

# 範例測試 (3): 複數共軛根
print("\n--- 複數共軛根範例 ---")
coeffs3 = [1, 0, 4]
print(f"方程係數: {coeffs3}")
print(solve_ode_general(coeffs3))

# 範例測試 (4): 複數重根
print("\n--- 複數重根範例 ---")
coeffs4 = [1, 0, 2, 0, 1]
print(f"方程係數: {coeffs4}")
print(solve_ode_general(coeffs4))

# 範例測試 (5): 高階重根
print("\n--- 高階重根範例 ---")
coeffs5 = [1, -6, 12, -8]
print(f"方程係數: {coeffs5}")
print(solve_ode_general(coeffs5))