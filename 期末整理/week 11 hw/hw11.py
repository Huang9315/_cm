import numpy as np

def dft(f):
    """
    執行離散傅立葉轉換 (Discrete Fourier Transform)
    對應圖片中的公式 1 (正轉換)
    
    參數:
    f : 輸入訊號 (一維陣列，代表 f(x))
    
    回傳:
    F : 頻域訊號 (一維複數陣列，代表 F(w))
    """
    f = np.asarray(f, dtype=complex)
    N = len(f)
    F = np.zeros(N, dtype=complex)
    
    # 直接根據定義實作雙重迴圈
    for k in range(N):
        sum_val = 0.0 + 0.0j
        for n in range(N):
            # Euler's formula: e^(-i * 2pi * k * n / N)
            exponent = -1j * 2 * np.pi * k * n / N
            sum_val += f[n] * np.exp(exponent)
        F[k] = sum_val
        
    return F

def idft(F):
    """
    執行離散傅立葉逆轉換 (Inverse Discrete Fourier Transform)
    對應圖片中的公式 2 (逆轉換)
    
    參數:
    F : 頻域訊號 (一維複數陣列)
    
    回傳:
    f : 重建的時間域訊號
    """
    F = np.asarray(F, dtype=complex)
    N = len(F)
    f = np.zeros(N, dtype=complex)
    
    # 直接根據定義實作雙重迴圈
    for n in range(N):
        sum_val = 0.0 + 0.0j
        for k in range(N):
            # 注意：逆轉換的指數是正的 (+i)
            exponent = 1j * 2 * np.pi * k * n / N
            sum_val += F[k] * np.exp(exponent)
        
        # 逆轉換通常需要除以 N (正規化)
        f[n] = sum_val / N
        
    return f

def verify_transform():
    """
    驗證函數：將函數 f 正轉換過去，再逆轉換回來，檢查是否等於原函數 f
    """
    print("--- 開始驗證 ---")
    
    # 1. 建立一個簡單的測試訊號 f
    # 這裡我們產生一個由兩個不同頻率的正弦波組成的訊號
    N = 8  # 取樣點數 (為了方便觀察，數值設小一點)
    x = np.arange(N)
    # f(x) = sin(x) + 0.5*cos(3x)
    original_f = np.sin(x) + 0.5 * np.cos(3 * x)
    
    print(f"原始訊號 f(x): \n{np.real(original_f).round(4)}") # 為了顯示整潔，只印出實部並取小數點
    
    # 2. 執行 DFT 正轉換
    F_omega = dft(original_f)
    print(f"\nDFT 結果 F(w) (前4項): \n{F_omega[:4].round(2)} ...")
    
    # 3. 執行 IDFT 逆轉換
    reconstructed_f = idft(F_omega)
    print(f"\nIDFT 重建訊號 f'(x): \n{np.real(reconstructed_f).round(4)}")
    
    # 4. 驗證兩者是否相等
    # 因為浮點數運算會有極微小的誤差，所以使用 np.allclose 來比較
    is_match = np.allclose(original_f, reconstructed_f)
    
    print("\n--- 驗證結果 ---")
    if is_match:
        print("成功！(Success): 原始訊號與經由 DFT -> IDFT 轉換回來的訊號一致。")
    else:
        print("失敗！(Fail): 訊號不一致。")

if __name__ == "__main__":
    verify_transform()