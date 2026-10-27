# MegaPreAttention 海光移植与 NV 审核

基准来源：[NV PR #1007](https://github.com/flagos-ai/FlagGems-vllm/pull/1007)，冻结提交
`89b0a5122a69d60f15afc552fb85b68dea8522d5`。海光开发基点 `0dfb5cd`。
2026-10-27，工作分支 `megapreattention_hygon`。NV 参考树只读，不将旧参考
`f060b50e` 视为当前 PR。仓库 workflow.md 底部 conv1d 模板与本任务无关。
目标仓库及其父目录未找到 optimization.md / deep_opt.md；按已有 workflow 的门槛执行，
性能未达标时在本文记录瓶颈证据、单变量试验与保留/回退决定。

## 编码前项目真相表（G1）

| 字段 | 内容 |
| --- | --- |
| Current op | 复合推理 API megapreattention；AllGather→gated QKV→Gemma RMSNorm→interleaved MRoPE→KV cache；无对应单一 aten schema |
| Editable files | 海光 backend ops/megapreattention.py、通用 ops 路由和导出、海光导出、tests/test_megapreattention.py、测试 helper、benchmark/test_megapreattention.py、core_shapes.yaml、pytest marker、工具和本文 |
| Read-only files | PR 提交五个文件、原始 NV 参考仓库、已生成数据包和 manifest、既有非本任务文件 |
| Build | Python/Triton JIT；bash -lc，保留 /opt/flagtree PYTHONPATH；compileall 检查新增文件；不重装整个环境 |
| Validation | tools/run_megapreattention.py --check-only；pytest tests/test_megapreattention.py；独立 torch oracle、cache、changed-input Graph、poison、布局与拒绝路径；分布式失败即停止 |
| Benchmark | tools/run_megapreattention.py；完整窗口 Graph，ms 越小越好；SpeedUp=baseline/candidate；预热20，交错100，独立5轮 |
| Active set | BF16；两模型真实 layer3；TP2/4；global tokens 4096/8192/16384/32768；16 case；额外小 shape/布局功能覆盖 |
| Aggregation | 每 rank 均值取 max，独立轮等权平均；16 case 的 speedup 算术平均；稳定性 (P90-P10)/median≤5% |
| Timeout | import/build 180s；单功能或 benchmark worker 600s；分布式 group timeout 180s；超时清理本任务子进程 |
| Profiler | 可用 hipprof --hip-trace --stats；先用 Graph 计时和 kernel 分解定位；不拿 profiler 中延迟作验收 |
| Autotune | 通信拓扑和同步不进行 per-rank 在线调优；初版离线 BM32/BN256/BK64、warps8/stages2；保留固定可复现候选；按证据再比较。NVIDIA tune config 不修改，本轮仅海光；协议相连 kernel 豁免在线 autotune，防止重放次数不一致导致死锁 |
| Fallback policy | NO_TORCH_COMPUTE_FALLBACK；生产只用 torch 元数据、empty 分配、无拷贝 view，SymmetricMemory 为仓库参考已用的非计算通信 runtime；拒绝未支持配置 |

## 对齐契约（G2）

| 字段 | 内容 |
| --- | --- |
| Torch API | 非单一 torch API；冻结 PR 的函数签名、独立 reference 和 checkpoint TP 布局；验证环境 torch2.9.0/HIP6.3.26045、Triton3.6.0 |
| Device/backend | 海光 BW/gfx936，80 CU；torch device 名 cuda；实际 P2P 与 SymmetricMemory rendezvous 需多卡探测 |
| Input contract | 连续 hidden[T/TP,H]、weight[2(localQ+localKV)*256,H]；head 内 Q/gate 交错，后接全 K、全 V；norm[256]；BF16 rope[P,64]；int64 positions[3,T]/slots[T]；所有 tensor 同设备；TP2/4；每 rank 正整数128倍 tokens；H≥256 且64整除 |
| Output contract | BF16 Q[T,localQ*256]、K/V[T,localKV*256]、gate[T,localQ,256]；gate 未激活；cache[blocks,localKV,block_size,512]原位写入，K在前V在后；不允许 cache 内部重叠和输入输出 alias |
| Dtype matrix | 仅BF16数据、int64索引；FP32/FP16/FP8、integer/bool/complex 计算不承诺支持，显式 NotImplementedError/ValueError |
| Promotion/acc dtype | BF16 dot→FP32累积→BF16投影；FP32 RMSNorm/Gemma(1+w)→BF16→FP32 NeoX RoPE→BF16；关闭 epilogue FMA 合并以保留参考舍入 |
| Semantics | 无 broadcast；三个非负section和32，eps有限非负；负slots忽略，其余必须唯一且在界内；positions必须在rope界内，生产不扫描值；不保证跨架构bitwise确定性；finite输入精度对拍，NaN/Inf按算术传播，signed-zero无额外保证 |
| Autograd | forward/inference only，requires_grad拒绝；不支持backward/sparse grad |
| Unsupported scope | TP1/8初版不覆盖、空tokens、非连续hidden/weight、其他D/R、混设备、重叠cache、Graph内首次分配/扩容；明确报错，无torch fallback |
| Reference-only torch use | 仅测试/benchmark中all_gather、matmul、norm、RoPE、cache拷贝、输入生成和哈希；生产不import这些模块 |

## 路径表（G3）

| 路径 | 触发条件 | 实现方式 | 配置 | 测试 | Benchmark | 风险 |
| --- | --- | --- | --- | --- | --- | --- |
| early error | 空/非法shape、dtype、device、layout、autograd | host元数据校验，在collective前报错 | 无 | 负向case | 不计收益 | rank需共同调用合法配置 |
| fast | 两模型16case | P2P对称内存推送AllGather + 系统scope屏障 + Triton dot和融合后处理；先正确后按测量推进tile重叠 | 离线固定策略 | 独立oracle、Graph、poison、changed-input | 完整窗口16case | 跨卡ordering、LDS/寄存器 |
| special | 非16字节对齐连续offset、strided positions/slots/cache | 普通指针load/store，不依赖TMA对齐；用显式stride | 同上 | 边界布局 | 单独记录 | cache非重叠 |
| general | 契约内其他H/head组合 | 同一masked dot路径，性能仅对active set验收 | 同上 | 小shape | 非验收收益 | 无全shape性能承诺 |
| unsupported | 其他dtype/TP/D/R/空输入等 | NotImplementedError或ValueError | 无 | 报错覆盖 | 不计 | 覆盖有限 |
| external runtime | 对称内存注册/映射/collective初始化与cleanup | PyTorch底层_SymmetricMemory（不计算结果）；不在计时内做host collective | 无 | 多卡P2P预检 | 初始化排除 | 必须所有rank同序调用/清理 |

## NV 审核范围

检查 Q/gate head 排列、KV复制、Gemma权重偏移、BF16中间舍入、RoPE轴选择、cache stride/负slot；
检查生产者发布、consumer acquire、旧buffer重用、Graph epoch、workspace扩容/释放；
检查 oracle 是否读取原始输入、测试覆盖、测量JIT、baseline范围及结果聚合。
问题只在有代码依据或复现后确认；硬件依赖、已公开契约限制和确定性bug分开记录。

## 已有证据与方案调整

- 海光为 BW/gfx936，80 CU；8卡可见。首次检查全卡利用率为0，设备间P2P可访问；
  测试先用物理0/1卡。共享机器每次扩卡/性能测量前重新检查占用。
- torch2.9 的 `_SymmetricMemory.empty_strided_p2p` 分配成功，但双卡 `rendezvous`
  实测报 `HIP invalid argument`。补齐 `enable_symm_mem_for_group` 后仍报错。
  因此 G1/G3 的外部通信 runtime 改用 **HIP原生IPC**，非torch计算fallback。
  torch仅提供未初始化allocation与本地view；通过 `hipMemGetAddressRange` 导出基址，
  `hipIpcGetMemHandle`/`hipIpcOpenMemHandle`交换映射，并保留子分配offset。
  相同allocator基址的IPC映射使用引用计数，防止多个workspace关闭同一映射。
  双卡映射、系统scope acquire/release屏障已通过独立探针。
- 原NV基线依赖SymmMem和vLLM，此环境不能原样运行。当前海光可运行基线定义为
  RCCL AllGather + rocBLAS matmul + torch.compile的Gemma norm/MRoPE/cache后处理。
  它没有AG/GEMM重叠，所有报告明确 `comparable_to_nv_native_baseline=false`，
  不将其speedup宣传为相对NV报告同级baseline的收益。
- 2026-10-27：35B/TP2/4096真实数据首次通过Q/K/V/gate/cache对拍、changed-input、
  poisoned payload与Graph replay；这是单case功能检查，不能代表16case或性能验收。

## NV 审核判断（持续更新）

1. **暂未确认数值算法错误。** Q/gate逐head交错、KV head复制、Gemma `1+w`、
   投影与norm的BF16舍入、NeoX局部旋转和cache负slot处理与冻结参考一致。
   海光对拍使用独立从原始shard开始的torch oracle，未读取候选scratch。
2. **测量前输入未恢复（可复现性问题）。** NV helper第461行将source.hidden依次乘
   -0.75/-1.25验证重放，第526行直接计时，中间未恢复。实际计时输入为两次BF16舍入后的
   约0.9375倍原hidden。baseline/candidate依然使用相同输入，因此这不证明其speedup无效，
   但按原seed/输入哈希复现时应注意。海光adapter在计时前恢复保存的原hidden。
3. **报告复现入口不完整。** NV benchmark调用run_distributed未传fixture_path，
   默认load_fixture生成随机权重；真实layer3必须通过外部调用显式注入。
   helper有fixture参数，但默认pytest/benchmark不能直接复现PR中的真实权重报告。
   海光adapter直接读取固定数据包并核对文件/packed-weight哈希。
4. **已披露的验证限制不当作bug。** PR只有10/16配置在所有五轮满足稳定性阈值；
   TP8与其他SM数没有硬件验证；同stream、所有rank同配置同序调用是协议前提。
   这些限制都保留在验收判断中。

NV审查来源：[冻结PR实现](https://github.com/Cheuring/FlagGems-vllm/blob/89b0a5122a69d60f15afc552fb85b68dea8522d5/src/flaggems_vllm/runtime/backend/_nvidia/hopper/ops/megapreattention.py)、[输入变更](https://github.com/Cheuring/FlagGems-vllm/blob/89b0a5122a69d60f15afc552fb85b68dea8522d5/tests/megapreattention_utils.py#L461)、[计时入口](https://github.com/Cheuring/FlagGems-vllm/blob/89b0a5122a69d60f15afc552fb85b68dea8522d5/tests/megapreattention_utils.py#L526)、[benchmark](https://github.com/Cheuring/FlagGems-vllm/blob/89b0a5122a69d60f15afc552fb85b68dea8522d5/benchmark/test_megapreattention.py)；
未向上游发布评论或改写其分支。

## 通信与组件试验记录

| Trial | 假设/改动 | 验证与证据 | 决定 |
| --- | --- | --- | --- |
| C0 | torch缓存allocation + HIP IPC + system-scope原子即可保证通信 | 小探针和35B单case可通过，但40MiB payload重复屏障/Graph会停顿，27B扩大测试复现 | 回退内存分配方案，相关功能结果仅作诊断，不视为验收 |
| C1 | HSA_FORCE_FINE_GRAIN_PCIE=1足以修正默认allocation | 重复屏障可推进但仍出现停顿 | 不依赖该环境变量作为修复 |
| C2 | 显式hipExtMallocWithFlags(...,hipDeviceMallocFinegrained=1)提供所需IPC可见性 | 40MiB双卡payload，5次普通调用+20次Graph重放，epoch连续到50，无停顿 | 保留；生产改用HIP runtime未初始化分配，torch不生成计算结果；完整矩阵重新验证 |
| P0 | BM32重复读取权重较多，增加BM可提高融合投影复用 | 隔离projection Graph实验，TP2/4K：35B BM32/64/128约1.083/0.667/0.499ms；27B约4.080/2.354/1.679ms；输出对拍通过；BM128共享内存65536B、253/254 registers、0 spills | 初选BM128/BK64/warps8/stages2；这是组件实验，不是完整窗口speedup，完整窗口重新验收 |
| C3 | 将payload放普通设备内存，仅flags使用fine-grained以恢复缓存性能 | 独立IPC区与系统屏障仍不足以保证普通payload跨rank可见；全功能测试出现大幅数值错误，4项失败 | **立即回退**，继续使用全fine-grained payload+flags；不放松精度/poison测试 |
| P1 | 分离投影/后处理可降低融合kernel的LDS和寄存器压力 | 128×128 GEMM、BK32/64、warps4/8实验：35B最佳约0.519ms、27B约1.631ms（含后处理）；未提供足够收益；hipprof确认GEMM约占90%、后处理约9% | 不合入此拆分方案，保留原融合路径，下一步依据HCU编译器实际MMAC参数测试 |
| P2 | HCU matrix_instr_nonkdim/kpack与更大GEMM tile可改善计算吞吐 | matrix_instr=32导致数百个spill、严重变慢；16/kpack2收益不稳定；256×256等拆分tile也慢于原融合kernel | 回退，未修改误差阈值/输入/基线 |
| P3 | 移植NV的tile-ready producer/consumer结构，重叠P2P与融合projection | 单独实验版本：每CU一个CTA，TP2保留4个producer、TP4保留8个；每128行由单producer写入并system-release epoch；consumer acquire后调用同一BF16计算helper；整调用前屏障保护旧payload复用。初步TP2边界与旧Graph扩容测试通过 | 完整功能与窗口测量中，未验收前不替换已验证生产版本 |

P3 后续结果：完整16 case及边界测试 **5 passed / 190.78s**，但TP2/4K窗口
测量candidate约1.89ms（35B）、6.55ms（27B），且baseline/candidate分布均有
明显波动。没有接受该不稳定性能样本，也不因结构接近NV而保留更慢的实现。
P3实验结束时生产回退到C2+P0分阶段路径；实验版本留在远端
`/data/lixichen/megapreattention_pipeline.py`，未作为支持的生产API合入。

最终功能回归在分阶段路径上补充了非capture扩容时立即释放旧workspace、capacity倍增、
非2的幂tokens，以及两组非默认MRoPE section/eps；结果 **7 passed / 264.37s**。
按20预热、交错100、每配置5轮的R0诊断矩阵已完成，均值SpeedUp约0.737，
7/16配置在所有轮次同时满足两实现稳定性阈值。其后发现测量adapter的初次调用
也执行了仅供功能测试使用的换卡检查，会建立额外设备上下文；现已限制为check_only。
共享机器随后出现新任务占用，R0及不稳定trial只保留为诊断，**不作为最终性能验收**。

新增host侧GPU进程监控，区分本任务进程与其他进程；后续测量使用实际空闲卡并记录
整个窗口的占用。用户要求等待空闲卡再测TP4，不能用有占用的卡补齐矩阵。
等待期间验证“fine-grained共享payload→Triton复制到本卡普通缓存→按row group投影”
的本地化路径；它与C3跨卡共享普通显存不同，跨卡通信仍只访问fine-grained区域。
本地缓存的group4与group8进行完整窗口隔离试验，均通过四个代表case精度。
group8相对group4的candidate延迟（35B/4K、27B/4K、35B/32K、27B/32K）
从0.690/1.975/4.828/14.412ms降至0.599/1.889/4.338/14.085ms；
整个窗口监控未发现其他任务进入选中的2/3卡。对应海光native比值约1.014/0.782/0.846/0.749，
仍不能作为完整active set验收。保留 **C2+本地缓存+BM128/warps8/GROUP_M8**。

局部投影试验进一步验证BM64/warps4的occupancy假设：35B约0.679ms、27B约2.307ms，
编译产生2–4个spill，慢于BM128/warps8/group8的0.468/1.558ms；回退小块方案。
最终实现使用5次设备launch（复用屏障、push、发布屏障、本卡cache复制、融合投影），
跨卡仍只访问fine-grained区域。普通缓存buffer仅在本卡读写，并随workspace generation保留。
最后一次全仓pre-commit已通过；最终本地缓存实现7项功能回归 **7 passed / 288.08s**，
使用物理1/2/3/5卡，整个功能窗口未发现其他任务进入测试卡。最终TP2/TP4各八个case均完成五轮，两个运行窗口的占用监控均无干扰。
最终状态为 `limited / needs deep_opt`：功能移植已验证，性能未通过验收。

细粒度allocation与设备指针通过HIP分配和Triton kernel访问；`_DevicePointer.fill_`
仅供诊断/测试污染缓冲区，内部为Triton store，不是torch compute fallback。
清理先同步并collective rendezvous，关闭所有远端IPC映射后再次barrier，再释放owner allocation。

## Deep Opt Board（由完整窗口性能触发）

2026-10-27：首轮35B/TP2/16K，candidate 2.4659ms、海光native 1.9313ms，
SpeedUp约0.783，低于0.9，触发deep optimization。普通IPC停顿已经修正，
不将原失败运行或组件实验混入此性能样本。

| 检查/策略 | 事实与下一步 |
| --- | --- |
| Metric | 同一Graph完整窗口，预热20、交错100；当前仅1轮诊断；慢rank均值，ms |
| Dispatch | 固定BM128/BN256/BK64，融合dot+norm/RoPE/cache；没有误走torch fallback |
| Config | 离线策略已冻结；不允许对collective独立在线调优 |
| Resource evidence | 融合kernel 253/254 registers、65536B LDS、0 spills；LDS已占满单CU容量 |
| 第一候选 | 将projection和后处理拆为两个Triton kernel；GEMM使用128×128 tile，减少CTA寄存器/LDS占用；保留精确BF16中间舍入和原后处理语义 |
| 对照方式 | 先隔离GEMM比较BK32/64、warps4/8；只选择通过精度的候选；再跑相同完整窗口active set，与冻结native比较 |
| 第二候选 | 若通信占主要剩余开销，考虑persistent tile-ready producer/consumer重叠；需重新证明驻留数量和每tile release/acquire；未验证前不作为默认实现 |
| 未改变项 | fixture字节、oracle、native后处理、计时/聚合和误差阈值全部冻结 |


## 集成与生产路径审计（G4/G7）

- API：`flaggems_vllm.megapreattention`、`clear_megapreattention_cache`；通过现有顶层自动导出接入。
  本算子没有单一aten schema，不增加torch dispatch registration；通用入口仅路由海光。
- 代码落点：海光backend、通用ops入口与两层导出；tests/benchmark同名入口，core_shapes完整16case。
- 在线调优豁免：仅海光离线调度，不改NVIDIA tune_configs；所有rank的协议与重放次数必须一致。
  最终固定BM128/BN256/BK64/GROUP_M8、warps8/stages2，依据上文对照试验选择。
- 真实数据：`tools/megapreattention_data/`提交6eb06d4；大张量不纳入Git。
- 静态检查：pre-commit --all-files全部通过，compileall与git diff --check通过。

| 生产调用 | 文件/范围 | 用途 | 审计结论 |
| --- | --- | --- | --- |
| torch.empty / empty_like | 海光_Workspace | 未初始化输出、epoch/peer元数据与本卡缓存分配 | 允许，无计算/拷贝 |
| torch.cuda.device / synchronize / is_current_stream_capturing | host路由、初始化、释放 | 设备上下文与生命周期协调 | 允许，非结果计算 |
| dist.get_rank / get_world_size / all_gather_object / barrier | IPC初始化与释放 | 交换地址/配置元数据，等待映射关闭后释放owner | 允许，计时不包含host初始化collective |
| Tensor shape/stride/dtype/device/is_contiguous/data_ptr/untyped_storage/view | host校验与gate view | 元数据检查及不搬移数据的view | 允许 |
| _DevicePointer.fill_ | 测试poison接口 | 内部调用Triton store，生产结果不依赖其调用 | 非torch compute |
| HIP C runtime | allocation/IPC | fine-grained未初始化allocation和跨进程映射 | 非外部计算库，无Torch fallback |

生产源码不导入tests/benchmark/fixture loader。数值结果由Triton完成；所有Torch matmul、norm、
输入复制与cache参考写入都位于测试/benchmark。审计机器记录为hy-accepted-torch-audit.json。

剩余边界：仅BF16、TP2/4、D256/R64、forward；其他组合显式拒绝。slots非负值必须唯一且在cache范围，
positions须在RoPE范围，生产不扫描值。所有rank须同配置同顺序、同一stream串行调用；Graph前预热，
clear前释放依赖Graph。捕获过的旧workspace在扩容后保留，直到collective clear，可能增加显存占用。
无法恢复单个rank提前退出的GPU轮询；调用方须保证分布式group存活并统一停止。


## 最终性能判定口径

- 五轮均保留全部100次有效采样，不用删除尾部、挑选轮次或改变输入提高比值。
- 每轮取各rank平均延迟的最大值，五轮等权平均后计算baseline/candidate。
- 速度阈值与稳定性分开：关键case SpeedUp至少0.9；每case的两实现、五轮都需满足span≤5%。
  即使所选卡全程无其他进程，也不把真实延迟波动改标为通过。
- TP4已经完成的结果为0.621–0.891，4/8 case全五轮稳定，全部速度低于0.9，
  因此当前性能状态为 **needs deep_opt**，不能标为完成性能验收。
- Deep opt停止原因：当前安全候选已覆盖内存可见性修复、本卡缓存、GEMM分离、MMAC参数、
  计算块资源占用和persistent通信重叠。保留通过功能验证且有代表case收益的本卡缓存方案；
  其余回退。进一步提升需要更有效的海光通信/计算重叠和MMAC调度，不能以Torch fallback、
  放宽误差或更换较弱baseline替代。


## 最终16case五轮结果（2026-10-27）

最终安全实现：fine-grained IPC payload/flags + 本卡cached staging + 分组融合投影。
TP4使用物理1/2/3/5卡，TP2使用1/2卡；host每2秒检查所选卡进程，两个完整窗口均未发现
其他任务进入，监控未报错。共享机器其余卡仍可能有任务，不声称整机独占。
每case独立worker五轮、每轮预热20及每实现交错100；全部48,000个rank样本保留。
80个case轮次精度与measurement_jit_free全部通过。

平均SpeedUp **0.7779**（16case等权算术平均），范围 **0.6214–1.0022**；
2/16速度达到0.9，其中35B/TP2/4K未通过全部五轮稳定性；10/16两实现在五轮均稳定。
**关键active set性能验收失败**。此基线不是NV的通信/GEMM重叠基线，不可直接比较NV speedup。

| Case | Native ms | Hygon ms | SpeedUp | 五轮均稳定 |
| --- | ---: | ---: | ---: | :---: |
| Qwen3.6-27B_tp2_t4096 | 1.4773 | 1.8912 | 0.781 | 否 |
| Qwen3.6-27B_tp2_t8192 | 2.8813 | 3.5938 | 0.802 | 是 |
| Qwen3.6-27B_tp2_t16384 | 5.4116 | 6.9865 | 0.775 | 是 |
| Qwen3.6-27B_tp2_t32768 | 10.5608 | 14.0189 | 0.753 | 是 |
| Qwen3.6-35B-A3B_tp2_t4096 | 0.6128 | 0.6115 | 1.002 | 否 |
| Qwen3.6-35B-A3B_tp2_t8192 | 1.0843 | 1.1366 | 0.954 | 是 |
| Qwen3.6-35B-A3B_tp2_t16384 | 1.9396 | 2.1977 | 0.883 | 是 |
| Qwen3.6-35B-A3B_tp2_t32768 | 3.6650 | 4.3360 | 0.845 | 是 |
| Qwen3.6-27B_tp4_t4096 | 0.9627 | 1.4393 | 0.669 | 否 |
| Qwen3.6-27B_tp4_t8192 | 1.8081 | 2.7692 | 0.653 | 是 |
| Qwen3.6-27B_tp4_t16384 | 3.3467 | 5.3540 | 0.625 | 是 |
| Qwen3.6-27B_tp4_t32768 | 6.5728 | 10.5778 | 0.621 | 是 |
| Qwen3.6-35B-A3B_tp4_t4096 | 0.4768 | 0.5351 | 0.891 | 否 |
| Qwen3.6-35B-A3B_tp4_t8192 | 0.7739 | 1.0056 | 0.770 | 否 |
| Qwen3.6-35B-A3B_tp4_t16384 | 1.4146 | 1.9548 | 0.724 | 否 |
| Qwen3.6-35B-A3B_tp4_t32768 | 2.6812 | 3.8347 | 0.699 | 是 |

原始文件：本地 `results/20261027/` 的hy-accepted-tp2/tp4.json、两份guard.json、environment.json、
functional.log、torch-audit.json；远端同名文件位于/data/lixichen。紧凑机器汇总为
performance-summary.json。较早R0记录已经移至本地diagnostics/，不得与本次结果混合。
环境与输入manifest、实现/基线/adapter SHA256都保留在environment和汇总中。

交付判定：**limited / needs_deep_opt**；生产无Torch计算回退；未接管既有vLLM执行路径，
仅提供显式调用API。功能移植与NV审查已落地，性能优化仍需后续工作。


最终集成补充验证：

- 标准pytest benchmark入口：单35B/TP2/4K，`--level core --mode kernel --warmup 1 --iter 1`
  与专用shape_file，**1 passed / 138.94s**。它只验证入口/常规kernel路径，单样本结果不纳入性能表。
- 两卡真实权重实际调用顶层`flaggems_vllm.megapreattention`与通用`flaggems_vllm.ops.megapreattention`，
  两个rank的Q/K/V/gate/cache均通过独立oracle对拍，collective cleanup正常。
- 最终带交付文档的暂存树再次pre-commit --all-files全部通过；源文件与正式计时环境清单SHA256一致。
