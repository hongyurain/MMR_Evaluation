import subprocess
from itertools import product
import shutil

# 超参数搜索空间
lr_list = [4.5e-4, 5e-4, 5.4e-3, 5.6e-3]
D_lr_list = [2.5e-4, 3e-4, 3.5e-4]
l2_alpha_list = [1.2e-2, 1.4e-2, 1.6e-2]
dataset = "baby"

best_score = -1
best_params = None
result=[]

for lr, D_lr, l2_alpha in product(lr_list, D_lr_list, l2_alpha_list):
    print(f"\nRunning: lr={lr}, D_lr={D_lr}, l2_alpha={l2_alpha}")

    cmd = [
        "python", "main.py",
        "--dataset", str(dataset),
        "--lr", str(lr),
        "--D_lr", str(D_lr),
        "--L2_alpha", str(l2_alpha)
    ]

    # 运行训练脚本
    process = subprocess.run(cmd, capture_output=True, text=True)
    output_lines = process.stdout.strip().splitlines()
    print(output_lines)

    if output_lines:
        last_line = output_lines[-1]
        print(last_line)
        if "best recall@20:" in last_line.lower():
            try:
                score = float(last_line.strip().split(":")[1])
                print(f"Best Recall@10 = {score}")

                # 更新最优参数和保存模型
                if score > best_score:
                    best_score = score
                    best_params = (lr, D_lr, l2_alpha)
                    print("New best! Saving model...")

                    # 保存模型文件为 best_model.pth
                    shutil.copy("model.pth", "./Data/"+dataset+"/best_model_text.pth")
                    result.append((lr, D_lr, l2_alpha, last_line))
            except ValueError:
                print("can not find numbers")
        else:
            print("can not find recall 10")
    else:
        print("None")


print("\n=== Best Setting ===")
if best_params:
    print(f"gen_lr={best_params[0]}, disc_lr={best_params[1]}, l2_reg={best_params[2]}")
    print(result[-1])
    print("Best model saved to: best_model.pth")
