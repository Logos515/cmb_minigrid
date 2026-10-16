## 一、方案 A：增加历史线索回忆损失（优先推荐）

你的任务真正需要的是：

\\[ (o_0,a_0,\ldots,o\_{12})\longrightarrow s\_{12} \\]

其中 \\(s\_{12}\\) 必须包含早期颜色信息。

所以可以给 RSSM 增加一个辅助训练目标：

\\[ \hat c_0=g\_\phi(s\_{12}) \\]

这里：

- \\(s\_{12}\\) 是 RSSM 在 handoff 的状态。
- \\(c_0\\) 是 t=0 观测中的 Key 颜色。
- \\(g\_\phi\\) 是一个小型分类头。
- \\(\hat c_0\\) 是模型对早期颜色的预测。

损失为：

\\[ \mathcal L\_{\mathrm{recall}} = \operatorname{CE}(g\_\phi(s\_{12}),c_0) \\]

整体训练损失变成：

\\[ \boxed{ \mathcal L\_{\mathrm{total}} = \mathcal L\_{\mathrm{RSSM}} + \lambda\_{\mathrm{recall}}\mathcal L\_{\mathrm{recall}} } \\]