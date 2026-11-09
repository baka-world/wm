# PR #8 A100 基线冻结记录

本分支从 `main` 的 `0364a63447b7ecc67ea1872ba938afb3536fefc1` 建立。PR [#8](https://github.com/lieuyimo/wm/pull/8) 当时尚未合并，其 head 为 `141500d1f69911ee2a7a9c04e9b932c6dcf971dc`。本轮不合并该 PR；而是把其报告和 19 个原始精简证据文件按字节复制到 [独立冻结目录](../examples/evidence/pr8-a100-baseline-20261109/)。原 `SHA256SUMS` 保持不变并全部校验通过；额外归档的 `PR8-REPORT.md` SHA256 为 `ea1b2dc8a686f1a7acb27e2310835c65e35fcaa826ab30870f8ee830608cf31d`。

校验：

```bash
cd examples/evidence/pr8-a100-baseline-20261109
sha256sum -c SHA256SUMS
sha256sum PR8-REPORT.md
```

PR #8 本身未提交 A100 原始检查点、完整数据和完整原始日志；当前环境也没有这些文件。因此本分支可以审查和校验 PR #8 已公开的基线汇总，**不能对其 GPU 权重重新预测，也不能将不同数据的结果直接当作严格配对改善**。本轮对照使用固定的新数据重新训练同一原始网络，并将新检查点和结果放在另一个目录；所有导出入口拒绝覆盖非空输出目录。
