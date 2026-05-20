# bushfire

根据 `Fire_Masks_n_Weather_composites.ipynb` 的逻辑，新增了 `reproduce_pdf_weather_nc.py`，用于在 **weather 输入为按年 NetCDF (`.nc`)** 时复现天气合成流程：

- 读取火情矢量（`FireNo`, `StartDate`）。
- 对每个 fire event 提取 `StartDate - 28天` 到 `StartDate` 的天气日数据。
- 按 `avg` 或 `sum` 聚合。
- 输出到固定 Sentinel-2 tile 范围与 20m 分辨率的 GeoTIFF。

## 依赖

```bash
pip install geopandas rasterio xarray pandas numpy
```

## 示例

```bash
python reproduce_pdf_weather_nc.py \
  --fire-shp /data/vector/Masks/FireHistory_2018_2020_UTM_T56HKG.shp \
  --weather-root /data/weather \
  --output-root /data/out \
  --tile T56HKG \
  --predictors ET Evaporation Max_temp Rel_Humid Sol_Rad Total_rain Vap_Press \
  --mode avg \
  --window-days 28 \
  --crs EPSG:32756
```
