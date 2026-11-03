# 上传 GitHub 与网站引用

本目录就是独立仓库根目录。包含代码、公开任务、地图、导航矩阵、离线评分器、
中英文说明、许可证和 GitHub Actions；无需依赖上级的原始工程。

## 本地检查

使用 Python 3.10+（macOS 系统自带的 Python 可能过旧）：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m dualworld validate
python -m unittest discover -s tests -v
```

Windows 使用相应的 Python 命令，并通过 `.venv\Scripts\activate` 激活环境。

## 建立 GitHub 仓库

在 GitHub 新建空仓库，建议名称 `DualWorldBench`，然后在本目录运行：

```bash
git init -b main
git add .
git commit -m "Release DualWorldBench evaluation code and datasets"
git remote add origin https://github.com/YOUR_GITHUB_OWNER/DualWorldBench.git
git push -u origin main
```

将 `YOUR_GITHUB_OWNER` 替换为实际账号或组织。提交使用你配置的 Git 作者信息。
`.gitignore` 会排除密钥配置、本地环境、缓存和运行输出。压缩包用于下载和存档，
GitHub 仓库应提交解压后的文件。当前没有单个超过 GitHub 100 MB 限制的文件。

代码延续已有公开版的 MIT 许可证，数据延续 CC BY 4.0；源码整理来源及对应关系见
[source-layout.md](source-layout.md) 和 [source-manifest.json](source-manifest.json)。

## 网站引用

仓库上传后，将网站 Code 按钮设为：

```text
https://github.com/YOUR_GITHUB_OWNER/DualWorldBench
```

建议在 GitHub 创建 `v0.1.0` Release，并将配套 ZIP 上传为下载附件；
网站 Download 按钮可以使用该 Release 页面：

```text
https://github.com/YOUR_GITHUB_OWNER/DualWorldBench/releases/tag/v0.1.0
```

网站静态托管只负责展示和链接下载。Python 评测通过本地命令运行，模型端点由
使用者配置。实际 GitHub 地址尚未指定，因此本版本保留上述可替换地址。
