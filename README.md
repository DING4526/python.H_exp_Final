# SparseTSF模型迁移与优化

## 项目概述

本项目涉及将SparseTSF模型从PyTorch框架迁移到MindSpore框架，并对其进行优化。SparseTSF是一种轻量级的长期时间序列预测（LTSF）模型，旨在通过最小化计算资源来有效建模时间序列中的复杂时间依赖关系。项目的主要目标是探讨模型迁移过程并对其性能进行优化，包括时间编码、偏差模块和超参数调节等改进。

> 原论文Github地址： https://github.com/lss-1138/SegRNN

## 主要特点

- **SparseTSF模型迁移**：成功将SparseTSF模型从PyTorch迁移到MindSpore。迁移过程中对数据处理、模型结构和训练过程进行了大量的代码重构。

- **模型优化**

	引入了多个优化方法，包括：

	- 时间编码：增强模型对时间序列中特征的捕获能力。
	- 偏差模块：进一步提升了模型的预测精度。
	- 超参数调节：优化了训练效率和性能。
	- 混合损失函数：平衡了不同类型的误差指标。

- **性能提升**：迁移后的模型在训练效率和预测精度方面均优于原PyTorch实现。

## 项目结构

- `data_provider.py`：负责数据加载和预处理。
- `models.py`：定义了SparseTSF模型的架构，包括各层、激活函数等。
- `exp_main.py`：管理训练、验证、测试和预测过程。
- `utils.py`：包含模型评估和结果可视化的工具函数。
- `requirements.txt`：项目所需的依赖项。

## 结果

- 迁移后，模型在ETTh1、AQ等数据集上展示了更低的损失和更高的预测准确度。
- 优化措施，尤其是时间编码和偏差模块的引入，显著提升了模型的性能。

## 未来改进

- 进一步探索如何处理非周期性的时间序列数据。
- 系统化地使用自动化超参数调优方法。
- 扩展模型应用到更多数据集，进行更广泛的评估。