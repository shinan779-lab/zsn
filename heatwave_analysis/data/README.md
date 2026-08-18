# 数据上传区

请把要分析的 ERA5 数据文件（NetCDF）和 PPT 里的图片（PNG/JPG）放在**这个目录**下，
然后告诉 Arena 里的助手"文件已上传"，它就能读取分析。

## 推荐命名（便于识别）

- 图片：`apr_tmax.png`、`apr_tmin.png`、`may_tmax.png`、`may_tmin.png`、
  `jun_tmax.png`、`jun_tmin.png`（或中文名也可，但英文名更稳妥）
- 数据：`era5_t2m_hourly_apr_jun.nc`（小时级）或 `era5_monthly_tmax_tmin.nc`（月平均）

## 上传方式（网页版，无需安装任何软件）

1. 打开 https://github.com/shinan779-lab/zsn
2. 右上角分支下拉框切换到 `arena/01a013ab-zsn`（能看到本目录结构）
3. 进入 `heatwave_analysis/data/` 文件夹
4. 直接把文件**拖进网页**（或点 "Add file" → "Upload files"）
5. 滚到底部，写个说明（如 "upload apr-jun images"），点 **Commit changes**

## ⚠️ 文件大小注意

- GitHub 单个文件限制 **100 MB**（超过 50 MB 会警告）。
- ERA5 **小时级**全量数据非常大（几十 GB 到几百 GB），**不适合直接传 GitHub**。
  建议只传：
  - 6 张图（很小，随便传）
  - 处理后的**月平均** NetCDF（通常几十~几百 MB，看分辨率）
  - 或只传欧洲区域子集 / 降分辨率（如 1° 或 2.5°）版本
- 如果文件实在太大，告诉助手，另想办法（网盘链接等）。
