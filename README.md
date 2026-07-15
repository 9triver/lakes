# Lakes

Lakes 是一个面向多区域遥感观测、水体标注和语义分割训练的本地 Web GIS。

系统将观测区域元数据库、本地影像、按需下载的 Sentinel-2 产品、外部水体数据和 U-Net 训练流程组织在同一个工作台中。一个观测区域可以包含多个水体，也可以与其他观测区域覆盖同一个水体。

## 功能

- 按省份浏览由本地影像目录定义的观测区域。
- 独立显示 OSM、HydroLAKES、ESA WorldCover、JRC GSW 和本地 Shapefile 标注。
- 查询并下载 Sentinel-2 SAFE/TCI 产品，指定观测区域当前使用的影像。
- 捕获当前地图视图和可见标注，记录训练区域。
- 检测重复或高度相似的训练视图。
- 为新增训练样本增量生成 Patch，并审核 include/exclude 状态。
- 按区域或全部区域训练 U-Net。
- 浏览历史训练任务、指标和模型权重。
- 随机选择观测区域进行模型验证，并将失败案例重新加入训练集。

模型预测只作为诊断图层记录，不会作为训练真值。

## 项目结构

```text
lakes/
  config/
    regions.toml                 区域定义和数据路径
  frontend/                      React + Vite + TypeScript 新前端
    src/
      app/                       应用入口、主题和客户端状态
      api/                       类型化 API 客户端
      features/                  按业务组织的观测区域、影像和训练功能
  src/lake_workbench/
    server.py                    依赖组装和服务入口
    http_handler.py              HTTP 请求上下文、路由调度和响应处理
    routes/                      按观测区域、训练、模型、Sentinel 等域组织的 API 路由
    catalog.py                   观测区域元数据加载、筛选、详情和摘要
    imagery/
      inventory.py               本地/下载影像库存、active 选择和产品登记
      raster.py                  TCI 渲染、影像拼接和模型预测矢量化
      rendering.py               观测区域 mosaic 和 XYZ 瓦片渲染编排
    sentinel/
      catalog.py                 Sentinel MGRS tile 匹配和产品覆盖率
      download.py                Copernicus 查询和下载
    models/
      metadata.py                模型权重发现、训练指标和持久化任务元数据
      validation.py              模型发现、推理缓存和随机验证
      unet.py                    U-Net checkpoint 加载和推理
    regions/
      config.py                  区域配置和标准数据路径
      service.py                 跨区域列表、训练数据和模型验证聚合
    training/
      catalog.py                 训练样本和 Patch 的区域级持久化操作
      identity.py                训练视图签名和范围相似度
      datasets.py                Patch manifest 和训练数据集摘要
      runner.py                  Patch 导出和 U-Net 训练任务适配
    water/
      annotations.py             site 水体标注的统一编排与结果契约
      providers.py               OSM/HydroLAKES/ESA/JRC/Local Label provider registry
      local_labels.py            本地 Shapefile 标注发现和 GeoJSON 转换
      layers.py                  ESA/JRC 栅格读取、多边形生成和缓存
    geo.py                       坐标转换、覆盖率和几何处理
    jobs.py                      下载、Patch 导出和训练后台任务
    paths.py                     项目根路径
    utils.py                     路径、CSV、参数解析和序列化工具
    static/                       React 构建产物，不纳入 Git
  scripts/
    prepare_data.py              下载公共基础数据
    build_site_metadata.py       构建观测区域元数据库
    site_metadata_sources.py     OSM、HydroLAKES 和 Sentinel 数据读取辅助
    export_training_patches.py   全量或增量生成 Patch
    train_unet.py                训练 U-Net
    download_sentinel.py         命令行 Sentinel 查询/下载
    precompute_*.py              预生成 ESA/JRC polygon
    migrate_site_data.py         将历史持久化数据迁移为 site 规范字段
  data/                          大型数据和模型，不纳入 Git
```

Python 服务默认在根地址提供构建后的 React 前端。

React 前端覆盖观测区域筛选、深链接、TCI 和 Tile 地图、外部及本地标注、Sentinel 产品查询下载、训练区域记录、训练样本管理、Patch 生成审核、模型训练和模型验证。

## 区域配置

区域统一配置在 `config/regions.toml`。当前包括：

- `gansu`：甘肃省，从本地 IMG 范围匹配外部水体。
- `shaanxi`：陕西省，从本地 IMG 范围匹配外部水体。
- `yunnan`：云南省，从本地 IMG 范围匹配外部水体。

`shared_data_dir` 配置与行政区域无关、只需保存一份的全球数据。目前包括 HydroLAKES 和 Sentinel-2 MGRS Tile Grid。

全局共享数据由 `regions.toml` 的 `shared_data_dir` 指定：

```text
data/shared/
  external_water/
    hydrolakes/
  sentinel_2_tiles/
```

每个区域使用统一的数据布局：

```text
data/regions/<region>/
  raw/
    local_imagery/<directory-id>/
    review/
    external_water/
      osm/
      esa_worldcover/
      jrc_gsw/
    sentinel_products/
  processed/
    site_metadata.gpkg
    site_metadata.csv
    esa_polygons/
    jrc_polygons/
    sentinel_products.csv
    active_imagery.json
    training_samples.csv
    training_labels/
    training_patches/
```

模型统一放在：

```text
data/models/<region>/<run>/
data/models/all/<run>/
```

每个训练目录通常包含 `config.json`、`history.json`、`manifest.csv`、`best.pt` 和 `last.pt`。

## 安装和运行

项目要求 Python 3.11 以上。

```bash
python -m venv .venv
.venv/bin/pip install -e .
PYTHONPATH=src .venv/bin/python -m lake_workbench.server --host 0.0.0.0 --port 18765
```

浏览器访问：

```text
http://127.0.0.1:18765
```

React 前端开发环境：

```bash
cd frontend
npm install
npm run dev
```

Vite 默认运行于 `http://127.0.0.1:5173/`，并将 `/api` 代理到 `18765`。生产构建会将 `index.html` 和 `assets/` 写入 `src/lake_workbench/static/`，随后由 Python 服务在 `http://127.0.0.1:18765/` 提供。生产构建执行：

```bash
cd frontend
npm run typecheck
npm run build
```

也可以使用：

```bash
PYTHONPATH=src HOST=0.0.0.0 PORT=18765 scripts/run_dev.sh
```

当前部署使用用户级 `lakes.service`：

```bash
systemctl --user restart lakes.service
systemctl --user status lakes.service
journalctl --user -u lakes.service
```

## 准备区域数据

下载公共基础数据并构建元数据库：

```bash
PYTHONPATH=src .venv/bin/python scripts/prepare_data.py --region gansu all
```

单独执行：

```bash
PYTHONPATH=src .venv/bin/python scripts/prepare_data.py --region gansu osm
PYTHONPATH=src .venv/bin/python scripts/prepare_data.py --region gansu hydrolakes
PYTHONPATH=src .venv/bin/python scripts/prepare_data.py --region gansu esa
PYTHONPATH=src .venv/bin/python scripts/prepare_data.py --region gansu jrc
PYTHONPATH=src .venv/bin/python scripts/prepare_data.py --region gansu sentinel-grid
PYTHONPATH=src .venv/bin/python scripts/prepare_data.py --region gansu metadata
```

需要代理访问 JRC Google Storage 时显式传入：

```bash
PYTHONPATH=src .venv/bin/python scripts/prepare_data.py \
  --region gansu \
  --proxy 192.168.30.107:7897 \
  jrc
```

Copernicus 查询和下载会显式忽略代理环境变量。凭据放在 `.env` 或环境变量中：

```text
COPERNICUS_USERNAME=...
COPERNICUS_PASSWORD=...
```

Sentinel SAFE/TCI 产品不会在 `prepare_data.py all` 中自动下载，由用户在界面或命令行按需选择。

HydroLAKES 和 Sentinel Grid 命令虽然保留 `--region` 参数以兼容统一命令格式，但输出均写入 `shared_data_dir`；任意 region 成功执行一次即可。

## 元数据库

观测区域身份由目录确定：`site_id = <region>_<directory-id>`。例如 `gansu_17407` 的显示名称为 `区域 17407`，当外部候选提供可信名称时显示为 `区域 17407（苏干湖附近）`；外部名称不参与身份判定。

手工重建某个区域的元数据库：

```bash
PYTHONPATH=src .venv/bin/python scripts/build_site_metadata.py --region gansu
```

`site_metadata.gpkg` 包含五个图层：

- `sites`：区域身份、显示名称、影像覆盖并集、日期、Tile 和候选统计。
- `site_coverage_core`：至少 80% 本地影像共同覆盖的核心区域。
- `imagery_assets`：每个 IMG 资产的有效覆盖、日期、分辨率、波段和路径。
- `local_label_features`：本地 Shapefile 中的全部水体要素。
- `external_water_features`：与区域相交的 OSM、HydroLAKES、ESA 和 JRC 水体候选。

单独预生成某个观测区域的 ESA/JRC 多边形缓存：

```bash
PYTHONPATH=src .venv/bin/python scripts/precompute_esa_polygons.py \
  --region gansu --site gansu_17407
PYTHONPATH=src .venv/bin/python scripts/precompute_jrc_polygons.py \
  --region gansu --site gansu_17407 --thresholds 50,75,90
```

## 训练工作流

1. 在观测区域页面选择本地影像或已下载的 Sentinel 产品。
2. 打开可信的 OSM、HydroLAKES、ESA、JRC 或本地标注。
3. 记录当前视图为训练区域。
4. 系统检测严格重复和视图范围高度重叠的相似样本。
5. 新样本自动按 `256 x 256`、stride `128` 增量生成 Patch。
6. 在 Patch 页面审核并设置 include/exclude。
7. 在训练页面按当前区域或全部区域启动 U-Net。
8. 在模型验证页面选择权重并随机验证观测区域。
9. 对预测较差的区域重新选择可信标注并补入训练集。

手工生成 Patch：

```bash
PYTHONPATH=src .venv/bin/python scripts/export_training_patches.py \
  --region yunnan \
  --patch-size 256 \
  --stride 128 \
  --preview-scale 2
```

只更新一个样本：

```bash
PYTHONPATH=src .venv/bin/python scripts/export_training_patches.py \
  --region yunnan \
  --sample-id yunnan_18292_dba0c430b7d0
```

命令会保留其他样本的 manifest 行以及已有 Patch 的 include/exclude 状态。

手工训练：

```bash
PYTHONPATH=src .venv/bin/python scripts/train_unet.py \
  --region all \
  --epochs 30 \
  --batch-size 8 \
  --device cuda
```

## API 约定

区域化 API 使用以下形式：

```text
/api/regions
/api/regions/<region>/sites
/api/regions/<region>/sites/<site_id>
/api/regions/<region>/sites/<site_id>/annotations/<source>
/api/regions/<region>/sites/<site_id>/local-labels
/api/regions/<region>/sites/<site_id>/imagery
/api/regions/<region>/training-samples
/api/regions/<region>/training-patches
/api/regions/<region>/training-runs
/api/regions/<region>/model-validation/models
/api/regions/<region>/model-validation/random
```

单个观测区域的详情、影像、标注和 Sentinel 操作必须使用实际 region。`all` 用于跨区域观测区域列表、训练样本、Patch、训练任务和模型验证等聚合场景。

OSM、HydroLAKES、ESA、JRC 和 Local Label 通过统一 annotation provider 读取。标注接口返回：

```json
{
  "site_id": "gansu_17407",
  "source": "jrc",
  "status": "available",
  "parameters": {"threshold": 75},
  "annotation": {}
}
```

`status` 为 `available`、`missing`、`empty` 或 `skipped`。JRC 使用 `?threshold=75` 指定阈值；本地标注先通过 `/local-labels` 列举，再使用 `/annotations/local?label_id=<id>` 读取。OSM、HydroLAKES、ESA 和 JRC 分别使用 `osm`、`hydrolakes`、`esa` 和 `jrc` 作为 `<source>`。

观测区域在 API、CSV 和后端领域模型中统一使用 `site` 与 `site_id`。旧 `/lakes` API 和 `lake_id` 字段不再受支持；已有持久化数据可执行以下命令完成迁移：

```bash
PYTHONPATH=src .venv/bin/python scripts/migrate_site_data.py
```

## 数据与 Git

`data/`、`attic/`、`.env` 和虚拟环境均被 Git 忽略。代码仓库不会同步大型影像、元数据库、Patch 或模型权重；这些数据需要通过独立的数据同步方案在机器间传输。

## 检查

```bash
PYTHONPATH=src .venv/bin/python -m compileall -q src scripts
PYTHONPATH=src .venv/bin/python -m pyflakes src scripts tests
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -p 'test_*.py'
(cd frontend && npm run typecheck && npm run build)
git diff --check
```
