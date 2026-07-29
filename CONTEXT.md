# Lakes Training Context

Lakes 将共享的基础观测数据与用户独享的训练工作空间分开，使不同训练方案可以复用观测数据而不共享逻辑 Patch 或训练产物。

## Language

**用户**:
一个具名的全局训练工作空间，定义逻辑 Patch 选择、各观测区域的来源变体选择，并拥有由此产生的训练数据、训练任务和模型。这里的用户不是登录账号或权限主体；内部接口和存储仍使用 Training Profile 标识。
_Avoid_: 配置, Dataset Config, 登录用户

**Training Sample**:
一次保存的观测影像、地图视图和来源标注快照；它是所有用户共享的训练事实。
_Avoid_: Profile Sample

**Logical Patch**:
从 Training Sample 影像划分出的固定 `512 x 512` 地理格网单元；底稿和审核状态均归属于一个用户。
_Avoid_: Training Patch, Included Patch

**Source Variant**:
某个观测区域中不可变的标注来源快照，身份包含来源类型、来源参数或本地标注资产以及几何内容。
_Avoid_: Data Source, Live Source

**Site Source Selection**:
用户为一个观测区域确认的 Source Variant 集合，用于为该区域的全部入选 Logical Patch 生成训练目标。
_Avoid_: Sample Source

**Dataset Config**:
将用户选择物化为训练 Patch 的版本化规则，例如输出尺寸、有效像元阈值和负样本比例。
_Avoid_: 用户, Training Profile

**Profile Dataset**:
由一个用户、训练范围和 Dataset Config 共同确定的可重建训练输入。
_Avoid_: Logical Patch Catalog

**Training Run**:
针对一个已冻结 Profile Dataset 执行的一次模型训练，完整保存输入清单和训练参数。

**Model**:
Training Run 产生的权重及元数据，归属于启动该 Run 的用户。

## Interface Invariants

- 根路径只负责选择和管理用户；工作区 URL 必须显式包含用户标识。
- 用户相关站点状态、Logical Patch 选择、Profile Dataset、Training Run 和 Model API 必须显式携带用户标识，不存在隐式默认用户。
- 区域目录、影像、标注和已有 Training Sample 是共享基础观测数据；Logical Patch manifest、预览、审核状态、Profile Dataset、Training Run 和 Model 均由用户独享。
- 保存一个已有或新增 Training Sample 时，只为当前用户生成该 Sample 的 Logical Patch；重建只处理当前用户已经拥有的 Sample，不会吸收其他用户的 Patch。
- 已归档用户不出现在 UI 中，深链接显示已归档错误；恢复能力只保留为后端管理接口。
