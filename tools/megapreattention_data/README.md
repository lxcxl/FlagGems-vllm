# MegaPreAttention layer3 测试数据

这组数据按飞书《10.26-MegaPreAttention&MegaMoE》的方法准备：真实 layer3
checkpoint 权重，加固定 seed 的合成 hidden。当前数据根目录：
`/data/lixichen/megapreattention_data/layer3_nv_20261027_v1`，宿主机与容器同路径。

## 内容与口径

- 两模型：Qwen3.6-27B、Qwen3.6-35B-A3B。
- 每模型 TP2/TP4 × 全局 tokens 4096/8192/16384/32768，共 16 case、48 份 rank hidden。
- 每模型 `layer3.pt` 含 q_proj、k_proj、v_proj、q_norm、k_norm 五类原始 BF16 权重，
  字典结构兼容 NVIDIA `load_fixture()`；q_proj 已包含每个 head 的 Q/gate，不额外生成 gate 权重。
- hidden 使用参考代码的 `20261010 + rank + global_tokens` 种子规则，
  `randn(..., dtype=bfloat16) * 0.05`，在海光 HIP 上一次生成后持久化。
- `shared/` 包含四组 int64 positions/slots 和一份 BF16 RoPE cache。
  RoPE 为 unscaled、theta=1e7、rotary_dim=64、NeoX、MRoPE sections=[11,11,10]、interleaved。
  按 vLLM 默认四倍扩展保留 1048576 行，当前只读取前 32768 行。
- 每个 rank 的 packed weight 在加载时由共享权重确定性切分、拼接并校验哈希。
  35B/TP4 下相邻 rank 共享 KV head，未把 KV heads 直接整除成零。
- KV cache 在加载时按 manifest 中的形状创建，block size=128，BF16，初值=0.375。
  这些常量缓冲区无需重复写盘。scale 为 FP32 ones(1)。
- manifest 记录 CUDA Graph、warmup20、交错各100采样和报告的每 case 轮数，
  此处只描述数据准备；算子验证结果见 docs/megapreattention_hygon.md。

`extraction.json` 保存源 checkpoint 路径、config/index SHA256、每个 tensor 的
key、分片、payload offset、形状和 SHA256；`manifest.json` 保存完整输入文件与张量哈希、
16 case 配置；`cases.csv` 为便于查看的矩阵。

## 提取、生成与校验

首次提取在宿主机执行，避免把整个模型拷入容器。输出目录必须不存在：

```bash
python3 tools/megapreattention_data/extract_layer3.py \
  --model-root /public-flash/models --output "$DATA"
```

容器内通过 `bash -lc` 加载环境；先检查 GPU 占用，再选择空闲卡：

```bash
HIP_VISIBLE_DEVICES=0 python3 tools/megapreattention_data/dataset.py prepare \
  --device cuda:0 --root "$DATA"
python3 tools/megapreattention_data/dataset.py verify --root "$DATA"
python3 tools/megapreattention_data/check_reference.py --root "$DATA" \
  --reference /data/lixichen/FlagGems-vllm-nvidia/tests/megapreattention_utils.py
```

宿主机再次核对原 checkpoint 的选中 payload，输出 source_validation.json：

```bash
python3 tools/megapreattention_data/extract_layer3.py --verify-source --output "$DATA"
```

以上命令前设置 `DATA=/data/lixichen/megapreattention_data/layer3_nv_20261027_v1`。
脚本拒绝覆盖已准备的数据。生成脚本和原始 BF16 payload 一并保留，便于复查。

## 两端加载同一份数据

```python
import sys
sys.path.insert(0, "tools/megapreattention_data")
from dataset import Dataset

data = Dataset("/data/lixichen/megapreattention_data/layer3_nv_20261027_v1")
inputs = data.load_rank(
    "Qwen3.6-27B_tp2_t4096", rank=0, device="cuda:0", group=process_group
)
# inputs.hidden/weight/positions/q_norm/k_norm/rope/cache/slots/scale/config/group
# 可以交给后续测试 adapter；所有加载、校验、分配必须放在计时与 Graph capture 之前。
```

`load_rank()` 返回测试用属性对象，已接入海光 tests/megapreattention_utils.py
及 tools/run_megapreattention.py。NVIDIA 参考仓库保持只读。
在 NVIDIA 机器上复制数据目录并用同一 loader 读取，才能复用相同的 hidden 和 RoPE 字节。
仅将 `layer3.pt` 传入现有 NVIDIA 入口仍会重新生成 hidden，不能视为相同输入回放。

NVIDIA 报告未公开原 checkpoint revision/hash 和实际 seed；目前固定的是海光现有
checkpoint 和参考代码的 seed 协议，不能声称与原报告输入逐字节相同。
不同 GPU 后端相同 seed 也不保证随机张量一致，因此比较时应加载本数据包。
容器未安装 vLLM，RoPE cache 使用其公开的 plain RoPE 公式生成并持久化；
原 vLLM 版本未核对，后续接入 baseline 时应直接复用该 cache。
