import math
import numpy as np

def section_1_probability_underflow():
    """
    第一部分：計算公平銅板連續投擲 10000 次皆為正面的機率。
    展示數值下溢 (Underflow) 與 Log 機率的應用。
    """
    print("\n" + "="*50)
    print("--- 1. 機率下溢與 Log(p^n) 計算 ---")
    print("="*50)
    
    p = 0.5
    n = 10000

    # 1. 嘗試直接計算 (會發生 Underflow，結果為 0.0)
    prob_direct = p ** n
    print(f"[直接計算] {p}^{n} = {prob_direct}")
    print("說明: 因為數值小於電腦浮點數精確度極限，故顯示為 0。")

    # 2. 使用 Log 計算: log10(p^n) = n * log10(p)
    # 使用 log10 可以直觀看到這是 10 的負幾次方
    log_prob = n * math.log10(p)
    print(f"[Log 計算] log10({p}^{n}) = {n} * log10({p}) = {log_prob:.4f}")
    print(f"結論: 機率約為 1 x 10^({int(log_prob)})，這是一個極小的數值。")


def section_2_entropy_metrics():
    """
    第二部分：計算熵、交叉熵、KL 散度、互資訊。
    並驗證 Cross Entropy 的性質。
    """
    print("\n" + "="*50)
    print("--- 2. 熵、交叉熵、KL 散度、互資訊 ---")
    print("="*50)

    # 定義分佈 (總和必須為 1)
    p = np.array([0.2, 0.5, 0.3]) # 真實分佈 (True Distribution)
    q = np.array([0.1, 0.8, 0.1]) # 預測分佈 (Approximate Distribution)
    eps = 1e-15 # 避免 log(0)

    print(f"分佈 P (真實): {p}")
    print(f"分佈 Q (預測): {q}")

    # 1. 熵 (Entropy) H(p)
    entropy_p = -np.sum(p * np.log2(p + eps))
    print(f"\n1. 熵 H(p): {entropy_p:.5f} bits")

    # 2. 交叉熵 (Cross Entropy) H(p, q)
    # H(p, q) = - sum p(x) log q(x)
    ce_pq = -np.sum(p * np.log2(q + eps))
    ce_pp = -np.sum(p * np.log2(p + eps)) # H(p,p) 其實就是 H(p)
    
    print(f"2. 交叉熵 H(p, q): {ce_pq:.5f} bits")
    print(f"   交叉熵 H(p, p): {ce_pp:.5f} bits (即 H(p))")

    # 驗證不等式: 吉布斯不等式說明 H(p,q) >= H(p,p)
    # 當 q != p 時，用錯誤的分佈編碼通常會消耗更多 bits，所以 H(p,q) 應該比較大
    if ce_pq > ce_pp:
        print(f"   [驗證結果] H(p, q) > H(p, p) 成立 ({ce_pq:.5f} > {ce_pp:.5f})")
        print("   說明: 使用錯誤的分佈 Q 來估計 P，會導致較大的交叉熵。")
    else:
        print("   [驗證結果] 數值異常 (理論上不應發生)。")

    # 3. KL 散度 (KL Divergence)
    # KL(p||q) = H(p,q) - H(p)
    kl_div = np.sum(p * np.log2((p + eps) / (q + eps)))
    print(f"\n3. KL 散度 D_KL(p||q): {kl_div:.5f} bits")
    print(f"   數值檢查: H(p,q) - H(p) = {ce_pq - entropy_p:.5f}")

    # 4. 互資訊 (Mutual Information)
    # 建立一個聯合分佈 P(X, Y)
    P_xy = np.array([[0.1, 0.2], 
                     [0.3, 0.4]])
    
    # 計算邊際分佈 P(x), P(y)
    P_x = np.sum(P_xy, axis=1)
    P_y = np.sum(P_xy, axis=0)
    
    mi = 0
    rows, cols = P_xy.shape
    for i in range(rows):
        for j in range(cols):
            if P_xy[i, j] > 0:
                # I(X;Y) = sum p(x,y) * log( p(x,y) / (p(x)p(y)) )
                mi += P_xy[i, j] * np.log2(P_xy[i, j] / (P_x[i] * P_y[j]))
    
    print(f"\n4. 互資訊 I(X; Y): {mi:.5f} bits (基於範例聯合分佈)")


def section_3_hamming_code_7_4():
    """
    第三部分：7-4 漢明碼編碼與解碼
    使用標準排列 (p1, p2, d1, p3, d2, d3, d4)
    """
    print("\n" + "="*50)
    print("--- 3. 7-4 漢明碼模擬 (編碼 -> 錯誤 -> 解碼) ---")
    print("="*50)

    # 原始資料 (4 bits)
    data = np.array([1, 0, 1, 1]) # d1, d2, d3, d4
    print(f"原始資料 (d1-d4): {data}")

    # --- 編碼 (Encoding) ---
    # 計算同位元 (Parity Bits)
    # p1 covers indices 1, 3, 5, 7 (binary xxx1) -> d1, d2, d4
    # p2 covers indices 2, 3, 6, 7 (binary xx1x) -> d1, d3, d4
    # p3 covers indices 4, 5, 6, 7 (binary x1xx) -> d2, d3, d4
    # 注意：這裡使用標準漢明碼位置 1-7
    # 位置: 1(p1), 2(p2), 3(d1), 4(p3), 5(d2), 6(d3), 7(d4)
    
    d1, d2, d3, d4 = data[0], data[1], data[2], data[3]
    
    p1 = (d1 + d2 + d4) % 2
    p2 = (d1 + d3 + d4) % 2
    p3 = (d2 + d3 + d4) % 2
    
    # 組合代碼字 (Codeword)
    codeword = np.array([p1, p2, d1, p3, d2, d3, d4])
    print(f"編碼後 (Codeword): {codeword}")
    print("位置對應: [p1, p2, d1, p3, d2, d3, d4]")

    # --- 模擬傳輸錯誤 (Error Injection) ---
    received = codeword.copy()
    error_idx = 2  # 0-based index 2 對應到第 3 個位置 (d1)
    
    print(f"\n[模擬] 翻轉索引 {error_idx} (第 {error_idx+1} 位元)...")
    received[error_idx] = (received[error_idx] + 1) % 2
    print(f"接收到的訊號:     {received}")

    # --- 解碼與校正 (Decoding) ---
    # 計算校驗子 (Syndrome)
    # s1 檢查位置 1, 3, 5, 7
    s1 = (received[0] + received[2] + received[4] + received[6]) % 2
    # s2 檢查位置 2, 3, 6, 7
    s2 = (received[1] + received[2] + received[5] + received[6]) % 2
    # s3 檢查位置 4, 5, 6, 7
    s3 = (received[3] + received[4] + received[5] + received[6]) % 2
    
    # 算出錯誤位置 (二进制: s3 s2 s1)
    error_pos = s3 * 4 + s2 * 2 + s1 * 1
    
    if error_pos == 0:
        print("結果: 傳輸無錯誤。")
    else:
        print(f"結果: 偵測到錯誤在位置 {error_pos} (Syndrome: {s3}{s2}{s1})")
        
        # 修正錯誤 (轉回 0-based index)
        corrected = received.copy()
        corrected[error_pos - 1] = (corrected[error_pos - 1] + 1) % 2
        print(f"修正後的訊號:     {corrected}")
        
        # 提取資料 (indices: 2, 4, 5, 6)
        decoded_data = np.array([corrected[2], corrected[4], corrected[5], corrected[6]])
        print(f"解碼資料 (d1-d4): {decoded_data}")
        
        # 驗證
        if np.array_equal(data, decoded_data):
            print("狀態: 資料復原成功！")
        else:
            print("狀態: 資料復原失敗。")

if __name__ == "__main__":
    section_1_probability_underflow()
    section_2_entropy_metrics()
    section_3_hamming_code_7_4()