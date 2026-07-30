# RePressDB JSS 大修复现实验包

仓库包含大修实验代码、冻结汇总结果、环境记录、数据筛选清单、轨迹清单，以及生成论文表格和数值宏的脚本。PEMS04、PEMS08 和 BDG2 原始数据不在仓库中重新分发。

普通 Git 仓库只放代码和小型结果。完整原始结果应作为 GitHub Release 附件上传，并将同一版本包存入 Zenodo 获取 DOI。

正式投稿前，在原实验电脑上运行：

```text
environment\capture_hardware.bat
```

脚本会生成 `hardware_exact.json`，其中包含 CPU 商品名和实验所在磁盘的型号，不记录硬盘序列号。
