# 实验与运行协议

## 环境与资源

使用外部命名Anaconda环境 `show-o-token-link`，不在项目内创建环境。激活：`conda activate show-o-token-link`。`environment/environment.yml` 为环境安装清单，包含Python、Conda库及pip依赖；`environment/environment.json` 记录基线运行时的硬件、软件和权重版本。

新电脑可用 `conda env create -n show-o-token-link -f environment/environment.yml`；导出包含CUDA 12.1 PyTorch wheel索引。该导出记录原Windows环境，不保证其他系统原样安装。数据及模型随项目复制，代码以自身目录定位资源。困难下载按AGENTS约定提供官方链接及校验信息，不自动长时间重试。

| 项目 | 256 | 512（默认） |
|---|---|---|
| checkpoint | `weights/showo/` | `weights/showo-512/` |
| 原始ID | 16×16 / 256个 | 32×32 / 1024个 |
| 官方模型 | `showlab/show-o` | `showlab/show-o-512x512` |

固定revision及权重校验在 `weights/manifest.json` 和 `weights/showo-512/manifest.json`。保留的官方源码来自commit `45a5a2de01d1ebd10cd5864d29310a76476cdf23`，未修改其运行源码；完整原始仓库位于归档。Show-o2和训练/生成示例已从工作目录移出。

RGB、短边bicubic缩放、中心裁剪、归一化[-1,1]。官方MAGVIT-v2 FP32输出0～8191的整数码，按行保存；采用FP32匹配原编码精度，未比较FP16编码。Show-o FP16及CUDA FP16 autocast；原码加50305进入扩充词表。`w_clip_vit=false`，不旁路连续视觉特征。

## 数据与固定任务

每个数据集共用 `images/`、`annotations/` 和 `manifest.json`；模型入口为 `encodings/showo-256/manifest.json` 或 `showo-512/manifest.json`，离散码在相邻 `ids/`。统一使用manifest.json，全量Pet旧文件名和200张清单已归档。现有资源已下载，正常验证不需要联网。

- Oxford-IIIT Pet：完整官方test 3669张，猫1183、狗2486，使用SPECIES字段，不测试37品种分类，也不将trainval作为测试。问题为 `Is the pet in the image a cat or a dog? Answer with only cat or dog.`。沿用原冻结肯定句/单词正则，主规则无效计错；另保留严格单词指标。旧200张/其余3469张分别计分。
- POPE：固定原版COCO POPE，500图、9000题，三个分组各3000题；官方问题追加 ` Answer with only yes or no.`。主规则要求开头yes/no，拒绝否定或矛盾；另记录官方解析。F1按汇总混淆矩阵计算，分组宏平均另标明。256原50图未超过80%门槛，全量为用户指定补测，未改写门槛。
- Imagenette：历史256固定100图，每类10图，严格别名计分，9%为早期诊断结果；512未测试。

batch=1、top_k=1、temperature=1、seed=0、max_new_tokens=16。没有通信干扰、训练或图像生成。模型训练不同，不能将两模型差值单独归因于分辨率。

## 运行入口

```powershell
python prepare_data.py verify --dataset oxford-pets --require-ids
python evaluate.py score --dataset oxford-pets
python evaluate.py verify --dataset pope --resolution 256
python prepare_data.py archive-verify
python evaluate.py run --dataset oxford-pets --output reports/showo-512/oxford-pets/new-run
```

`prepare`从本地共用清单生成尚不存在的模型清单，拒绝覆盖现有清单；Pet缺少的测试图可从官方本地压缩包提取。`verify`检查图像哈希及已保存ID。新增模型尺寸的ID由evaluate的encode阶段生成。

evaluate的 `run`依次调用两个独立进程：`encode`、`answer`。编码进程读原图；回答进程安装图像访问拦截，仅读ID与问题，禁止导入MAGVIT。需要链路核验时，手动执行 `python evaluate.py reference --dataset oxford-pets --output reports/showo-512/oxford-pets/new-run`，数据集、分辨率、输出目录及子集参数须与对应运行一致。参考进程重新编码原图，对照已保存的完整生成结果，默认最多抽查6条，Pet优先采用固定参考样本。单阶段也可分别调用。`--limit N`用于新运行的小批诊断并写入协议，不影响正式完整基线范围。`score`/`verify`读取已保存回答，不加载模型、不改写基线。

新运行必须指定新 `--output`；存在metrics.json的正式基线禁止重新编码/回答覆盖。中断的新运行可在冻结配置不变的情况下续跑；已有JSONL回答按key复用，提示词和包哈希再次检查。协议冻结清单、代码与版本哈希。

单图仍使用 `python link.py encode`、`answer`、`reference`、`compare`；参数帮助见 `python link.py --help`。它们默认512，256必须显式 `--resolution 256`。保存ID后answer只依赖ID、问题和必要格式。源码默认离线加载本地权重；未指定HF_HOME时，库的少量缓存写入系统临时目录，不在项目内创建缓存。

## 当前证据与历史

每模型只有一份 `reports/showo-*/REPORT.md`。每个已测试数据集保留 `answers.jsonl`（每行record_id及完整原始record）、`results.json`（逐题标签和评分）、`metrics.json`（原汇总、冻结协议、环境、阶段性能、路径抽查、审计和错误行索引）。重复POPE图题共享同一record_id，正式统计仍按9000题。

原始回答值、干净ID和标签未改变；所有旧报告、日志、一次性脚本、旧清单及迁移文件原字节在 `archive/baseline-history.zip`。`archive/manifest.json`保存原相对路径、SHA256、大小和当前位置；旧迁移入口 `reports/project/layout-migration/manifest.json`指向此清单。逐文件核验后才删除旧文件。

性能表计时含加载，编码可复用已有ID；整卡显存采样包含其他进程。256 Pet曾并发做两次短诊断，6574MiB不是单模型独占峰值，不能据此比较两模型速度或独占显存。原记录及说明完整保留在metrics和归档。公开预训练重叠未审计，错误原因尚需消融验证。
