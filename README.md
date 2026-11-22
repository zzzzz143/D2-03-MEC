# D2-03 最小熵耦合实验复现

复现 ICLR 2023 论文 *Perfectly Secure Steganography Using Minimum Entropy Coupling* 的 GPT-2 top-40 实验。主实验固定 80 位均匀随机消息、20 位分块和 0.1 bit 后验熵终止阈值，逐条独立解码后对比 80 位。

## 文件与输入

将三个 ZIP 解压到同一个目录，保留以下层级。

```text
README.md
coupling.cpp
imec.py
experiment.py
prepare_prompts.py
validate_coupling.py
requirements.txt
assets/
  gpt2/                      # GPT-2 权重、tokenizer 与配置
  wiki-parquet/              # WikiText-103-v1 的两个训练分片
  prompts-1000.jsonl         # 按固定种子抽取的 1,000 个提示
  SHA256-gpt2.txt
  SHA256-wikitext.txt
results/
  pilot-20/
  cpu-smoke-20/
  formal-20/
```

数据和模型可以离线加载。只复跑固定评测批次时，运行入口读取 `prompts-1000.jsonl` 和 GPT-2 目录。两个 WikiText-103 分片用于重建提示集。源文件版本如下。

- GPT-2 为 `openai-community/gpt2`，提交 `607a30d783dfa663caf39e06633721c8d4cfcd7e`。提供未经微调的 FP32 Safetensors 及分词文件。 [模型卡](https://huggingface.co/openai-community/gpt2/tree/607a30d783dfa663caf39e06633721c8d4cfcd7e)
- WikiText-103 为 `Salesforce/wikitext` 的 `wikitext-103-v1`，提交 `b08601e04326c79dfdd32d625aee71d232d685c3`。使用分词版训练分片抽取上下文。 [数据集文件](https://huggingface.co/datasets/Salesforce/wikitext/tree/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-103-v1)

## 运行

已验证环境为 Linux x86-64、Python 3.10.20、GCC 11.4、PyTorch 2.2.0+cu121 和 Transformers 4.40.1。Windows 上可在 WSL2 的 Linux 环境运行。选择 Python 3.10，安装适合本机的 PyTorch 2.2.0 和以下依赖。CUDA 12.1 的安装命令如下。CPU 环境使用 `cpu` 索引并在实验命令中指定 `--device cpu`。

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.2.0
python -m pip install -r requirements.txt
sha256sum -c assets/SHA256-gpt2.txt
sha256sum -c assets/SHA256-wikitext.txt
g++ -O3 -std=c++17 -fPIC -shared coupling.cpp -o coupling.so
python validate_coupling.py
```

可选复核提示集抽样过程。脚本读取两个训练分片，恢复空行，按原程序的一级标题规则切分文章。在至少有 1,024 个字符的正文中，无放回抽取 1,000 篇，保留各篇的前 1,024 个字符。抽样种子为 20261119。

```bash
python prepare_prompts.py assets/wiki-parquet rebuilt-prompts.jsonl
cmp rebuilt-prompts.jsonl assets/prompts-1000.jsonl
```

正式运行命令如下。输出到新目录，保留已随包提供的实测结果。运行时建议用 `CUDA_VISIBLE_DEVICES` 选择空闲卡并将可见设备编号传给 `--device`。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -u experiment.py \
  --model assets/gpt2 --prompts assets/prompts-1000.jsonl \
  --device cuda:0 --batch-size 16 --workers 16 --out rerun/formal-20
```

短跑可指定 `--limit 8 --batch-size 8 --workers 8 --out rerun/pilot-20`。CPU 短跑可用 `--device cpu --batch-size 2 --workers 2`。默认每个批次的编码和解码阶段各有 180 秒超时上限，另有 512 token 上限。不同设备、数值库与批大小可能导致个别轨迹变化，比较时应同时核对配置和总体指标。

## 结果字段

- `trajectories.jsonl` 每行一条轨迹，记录原始与恢复的位串、生成 token ID、状态、信道熵和逐条效率。解码从同一提示和收到的 token ID 开始，重新运行 GPT-2，并独立初始化后验。
- `summary.json` 汇总尝试数、编码成功数、解码完成数、逐位核验通过数、超时、错误、其他未完成数、误码和设备信息。`not_completed` 指未达到编码终止条件的轨迹，可能与超时计数重叠。`status` 中的 `ok`、`error`、`timeout`、`incomplete` 互斥。
- 单条编码效率为编码速率 80 / token 数除以同轨迹的平均信道熵，等于 80 除以该轨迹每步的 top-40 信道熵总和。`mean_efficiency` 为所有编码完成轨迹的单条效率均值，验收线为 0.744。`pooled_efficiency` 是总消息位数除以总信道熵，作为汇总对照。

## 方法来源

依据论文第 5 节和作者公开代码实现 4 个离散后验的迭代更新。稀疏后验使用正质量边，避免为两个边缘分布补齐后分配方形稠密矩阵。C++17 子程序实现同一种逐步配对最大剩余质量的贪心耦合，小规模核验采用独立稠密参考。原仓库用于核对实验配置，提交代码可直接运行并使用本地模型，无需日志平台账号。

- [ICLR 2023 论文](https://arxiv.org/html/2210.14889#S5)
- [作者实验仓库](https://github.com/schroederdewitt/perfectly-secure-steganography)
- [贪心耦合算法来源](https://arxiv.org/abs/1611.04035)
