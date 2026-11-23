# hybrid：多辆车拼成一辆车的 3D 模型

目标：车头来自 A 车、车身来自 B 车、车尾来自 C 车，生成一辆高精度的混合车 3D 模型，之后再转成可以实际拼搭的乐高模型（放大比例，不限于 Speed Champions 的 8 宽）。

这是新路线的第 1 步：**先用现成的 3D 模型（ShapeNet）验证"裁切 + 接缝融合"的效果**，不涉及照片重建。

## 流程

1. **单车实体化**（`solid.py`）：ShapeNet 网格不封闭（缺面、单层玻璃、带车内零件），所以用深度融合得到实体。具体做法是从 96 个方向（包括车底）渲染深度图，任何相机看到"在表面前方"的体素都标成空，剩下的就是实体。这样能补上破洞、去掉车内零件，并保留所有从外面看得见的凹处（轮拱、皮卡货箱）。默认 1.2 cm 体素，然后转成 SDF。
2. **统一坐标系**：单位是米，x 朝前、y 朝左、z 朝上，地面在 z=0。每辆车按车身宽度（默认 1.95 m）统一尺度。前后轴从车底轮廓自动检测（轮胎接地，两轴之间的车底不接地）。
3. **按轴定位拼接**（`compose.py`）：以车身那辆车的坐标系为准。车头按前轴对齐，车尾按后轴对齐，所以车头和车尾都保留自己的轮子、轮拱和悬伸。接缝默认在前轴后方 0.75 m、后轴前方 0.75 m，避开轮拱。
4. **截面匹配**：车头和车尾的高度、宽度从远端到接缝逐渐缩放，到接缝处对上车身的截面（`--match`，0 表示不变形，1 表示完全对齐），避免矮车头直接撞上高驾驶室。
5. **接缝融合**：在过渡带（`--band`，默认 0.35 m）内插值 SDF，形状会逐渐变形过去。过渡带中间再混入两者的并集（`--fill`），防止截面差别大时中间被"掐细"，比如货箱接发动机盖的情况。
6. **颜色**：每个顶点到来源车"看得见的表面"上取最近点的颜色，过渡带内按权重混合。

## 用法

需要 bricknet 环境（有 torch/CUDA、scikit-image、trimesh），在仓库根目录运行：

```bash
python -m hybrid search veyron                      # 按名字找 ShapeNet 车
python -m hybrid source <id>                        # 单车实体化 + 预览（结果会缓存）
python -m hybrid compose --front <id> --body <id> --rear <id> --name my_car \
    [--front-cut 0.75 --rear-cut 0.75 --band 0.35 --match 0.8 --fill 1.0]
```

车头朝后的模型加 `:flip`，例如 `--rear 8b22da6bba6c81452af3f274e91bb46f:flip`。

输出在 `outputs/hybrid/<name>/`：

| 文件 | 内容 |
|---|---|
| `sources.jpg` | 三辆来源车，没用到的部分是灰色 |
| `layout.png` | 侧面布局：每辆车的顶部轮廓、接缝过渡带、车身的两根轴 |
| `hybrid.jpg` | 混合车四个视角 |
| `turntable.gif` | 360° 转台 |
| `hybrid.glb` / `hybrid.ply` | 带顶点颜色的网格 |
| `summary.json` | 参数、接缝位置、每段的平移量和缩放比例 |

## 例子（单次约 7 秒，来源车缓存之后）

```bash
python -m hybrid compose --front e6ec389325de7ae5becf71e2e014ff6f --body d474f5971b5e8ecae9795e5d56b7e9f \
    --rear 29793366c4b45bf6f43c51e53897fea --name fordgt_tacoma_veyron      # 福特 GT + 塔科马皮卡 + 威龙
python -m hybrid compose --front 29793366c4b45bf6f43c51e53897fea --body e6ec389325de7ae5becf71e2e014ff6f \
    --rear d474f5971b5e8ecae9795e5d56b7e9f --name veyron_fordgt_tacoma      # 威龙 + 福特 GT + 皮卡货箱
```

## 已知局限和下一步

- 尺度按车宽统一，没有用真实尺寸。可以加一个"已知车长"的选项。
- 接缝只能沿 x 方向切成三段，还不能按语义部件切（比如只换车灯或尾翼）。
- 过渡带下沿有时会多出一小片"裙边"，因为并集把两车的底边都保留了。
- 下一步：把单车 3D 换成"照片 → 3D"（图像到 3D 生成模型，或者检索 + 轮廓校正），然后做乐高化。
