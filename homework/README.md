# 简陋版 MNIST 实验

不依赖现成训练库的超简陋实现，方便当作作业展示。包含：
- 手动下载并解析 MNIST 的 idx 数据。
- PCA + QDF、KNN、LDA + QDF 三种模型的小样本实验。

## 使用
1. 创建虚拟环境（可选）：
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
2. 安装依赖：
   ```bash
   pip install -r homework/requirements.txt
   ```
3. 运行脚本（默认缓存到 `~/.cache/mnist_basic`）：
   ```bash
   python homework/mnist_basic.py
   ```

运行后会下载 MNIST，并在 4000 张训练图和 500 张测试图上给出三种方法的准确率。
