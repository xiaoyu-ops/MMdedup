from tqdm import tqdm
import time

# Short smoke test for ASCII tqdm and non-ASCII console output.
print("开始中文输出测试：")
for _ in tqdm(range(100), desc="分类中", ascii=True):
    time.sleep(0.01)

print("分类完成，测试中文输出")
