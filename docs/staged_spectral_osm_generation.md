# 分阶段生成光谱∩OSM训练数据

所有命令在 `/home/lake/lakes` 执行。只更新脚本不会自动启动省级任务。

## 推荐：先补齐种子，再批量生成Patch

```bash
PYTHONPATH=src .venv/bin/python -u scripts/generate_spectral_osm_training_data.py \
  --region hunan --all-sites --stage seeds --continue-on-error
```

该命令扫描每个有本地影像的Site的**当前活动期次**，匹配完整身份的种子缓存直接复用；不会将“有Patch”误当作“有种子”，也不生成所有历史期次。缺失或不匹配的缓存才生成。缓存身份沿用现有算法、影像签名、范围、分辨率等规则。

种子补齐后：

```bash
PYTHONPATH=src .venv/bin/python -u scripts/generate_spectral_osm_training_data.py \
  --region hunan --all-sites --stage patches --batch-sites 20 --continue-on-error
```

Patch阶段只接受匹配的已有种子，不会隐式下载/生成缺失种子。先保存样本，再每20个待处理Site合并为一次Patch提交；已存在同样尺寸/步长且预览完整的样本跳过。排除状态不会因“跳过”而恢复。生成时保留仍在当前清单中的人工审核字段；之前并发覆盖丢失的历史人工状态不能凭脚本推断恢复。

需要在Patch阶段成功后进一步构建已纳入Patch的训练NPZ，显式加 `--build-dataset`。默认只生成网页逻辑Patch，不自动开始训练、不自动统一排除低于1%的Patch。

也可一次执行两个阶段：`--all-sites --stage all --batch-sites 20`。旧的 `generate_missing_site_training_data.py --region hunan --first 400` 仍兼容，但仅选择没有Patch记录的Site；全面核查/修复建议使用 `--all-sites`。

## 断点与预览

日志首部打印状态文件路径。中断后使用：

```bash
PYTHONPATH=src .venv/bin/python -u scripts/generate_spectral_osm_training_data.py \
  --region hunan --resume /完整路径/任务状态.json --stage all --continue-on-error
```

恢复原Site列表，不会因已有数据增加而把`--first`换成下一批。仍会检查实际缓存和Patch完整性，不能只信任状态记录。自定义尺寸参数恢复时需保持一致。单Site失败可记录后继续；失败批次会逐Site隔离重试。全成功返回0，有失败返回1，重复生成任务被拒绝返回2。

只检查不生成：添加 `--dry-run`。该模式不写任务状态、种子、样本或Patch；仅检查缓存选择。

## 并发保护及限制

同工作区/省份CLI生成任务拒绝重复启动。Workspace构建和提交使用可重入跨进程文件锁，网页审核及脚本共享元数据锁；更改审核清单采用原子CSV发布。目录切换捕获主动中断后恢复旧目录。历史已经运行的旧Python进程不会热加载新锁，发布前必须确认旧任务已停止，并重启网页后端。

不要让两个任务使用同一个 `> docs/hunan01.log`：Shell会在脚本获得锁之前截断日志。后台运行建议单独日志，例如 `docs/hunan_seeds_20261029.log` 和 `docs/hunan_patches_20261029.log`。脚本日志强制flush，使用`-u`便于观察。

只修复当前工作流，不自动清理临时目录，不自动恢复未知的丢失记录，不自动修改训练/验证/测试划分。批量提交仍保留现有目录发布机制，每批复制旧预览一次；不是完全零复制。
