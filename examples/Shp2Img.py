import os
import numpy as np
import geopandas as gpd
import rasterio
import logging
import matplotlib.pyplot as plt
from rasterio import features
from rasterio.transform import from_bounds
from matplotlib.colors import ListedColormap
from typing import Union, Tuple, Optional, Dict, Any
from shapely.geometry import box
from PIL import Image
from pathlib import Path

FORMAT_TO_DRIVER = {
    'tif': 'GTiff',
    'png': 'PNG',
    'jpg': 'JPEG',
    'jpeg': 'JPEG',
    'bmp': 'BMP'
}

class SHP2IMAGE:
    """
    将shp文件转换为与遥感影像像素对齐的图像标签
    """
    
    def __init__(self, mode: str = 'reference', format: str = 'tif', log_level: int = logging.WARNING):
        """
        Parameters:
        -----------
        mode    转换模式: 'reference' - 参考遥感影像元数据; 'custom' - 自定义转换参数
        format  输出图像格式
        """

        if mode not in ['reference', 'custom']:
            raise ValueError("模式必须为 'reference' 或 'custom'")
        if format not in FORMAT_TO_DRIVER:
            raise ValueError(f"不支持的图像格式: {format}, 可选值: {list(FORMAT_TO_DRIVER.keys())}")
        self.mode = mode
        self.format = format
        self.driver = FORMAT_TO_DRIVER[format]

        self.convert = None
        if mode == 'reference':
            self.convert = self.convert_reference
        elif mode == 'custom':
            self.convert = self.convert_custom

        self.log_level = log_level
        self.logger = logging.getLogger(f"{self.__class__.__name__}_{id(self)}")
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        self.logger.propagate = False   # 防止日志传播到根logger
        handler = logging.StreamHandler()
        handler.setLevel(log_level)
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(log_level)

        


    def _convert_common(self, gdf: gpd.GeoDataFrame, 
                       transform: rasterio.transform.Affine, 
                       shape: Tuple[int, int],
                       crs: rasterio.crs.CRS,
                       column: str = 'class',
                       fill_value: int = 0,
                       dtype: type = np.uint8) -> np.ndarray:
        """
        Parameters:
        -----------
        gdf         包含几何图形和标签的GeoDataFrame
        transform   输出图像的仿射变换
        shape       输出图像的形状
        crs         输出图像的坐标参考系统
        column      用于提取标签值的列名，如果为None则所有要素使用相同值
        fill_value  背景填充值
        dtype       输出数组的数据类型
            
        Returns:
        --------
        label_image 转换后的标签图像
        """
        # 确保GeoDataFrame的CRS与目标CRS一致
        if gdf.crs != crs:
            self.logger.info(f"将SHP的CRS从{gdf.crs}转换到{crs}")
            gdf = gdf.to_crs(crs)

        # 准备几何图形和值的列表
        if column and column in gdf.columns:
            shapes = [(geom, value) for geom, value in zip(gdf.geometry, gdf[column])]
        else:
            if column:
                self.logger.warning(f"列 '{column}' 不存在于GeoDataFrame中，所有要素值将置为1。")
            shapes = [(geom, 1) for geom in gdf.geometry]
        
        # 使用rasterio的rasterize函数进行转换
        label_image = features.rasterize(
            shapes=shapes,
            out_shape=shape,
            transform=transform,
            fill=fill_value,
            dtype=dtype,
            all_touched=False  # 所有接触到的像素不被标记
        )
        
        return label_image
    

    def _save_image(self, image: np.ndarray, 
                    output_path: Path,
                    transform: rasterio.transform.Affine,
                    crs: rasterio.crs.CRS,
                    dtype: type = np.uint8,
                    fill_value: int = 0) -> Dict[str, Any]:
        """
        将图像保存到文件的通用方法。

        Parameters:
        ----------
        image       将要保存的图像
        output_path 保存图像的路径
        transform   图像的仿射变换
        crs         图像的坐标参考系统
        dtype       图像的数据类型
        fill_value  nodata时的填充值
        """
        output_path = output_path.with_suffix(f".{self.format}") # 修改后缀名为指定格式
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        metadata = {
            'driver': self.driver,
            'dtype': dtype,
            'nodata': fill_value,
            'width': image.shape[1],
            'height': image.shape[0],
            'count': 1,
            'crs': crs,
            'transform': transform
        }
        
        with rasterio.open(output_path, 'w', **metadata) as dst:
            dst.write(image, 1)
        
        self.logger.info(f"图像保存路径: {output_path}")
        return metadata
    



    def convert_reference(self, 
                         shp_path: Union[str, Path],
                         reference_image_path: Union[str, Path],
                         output_path: Optional[Union[str, Path]] = None,
                         column: Optional[str] = 'class',
                         fill_value: int = 0,
                         dtype: type = np.uint8,
                         **kwargs) -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
        """
        参考遥感影像元数据进行转换
        
        Parameters:
        -----------
        shp_path    shpfile文件路径
        reference_image_path    参考遥感影像路径
        output_path 输出图像路径，如果为None则不保存
        column      用于提取标签值的列名
        fill_value  背景填充值
        dtype       输出数组的数据类型
        **kwargs    其他参数传递给rasterio.open
            
        Returns:
        --------
        label_image 转换后的标签图像
        metadata    如果output_path不为None，返回元数据字典
        """

        shp_path, ref_img_path = Path(shp_path), Path(reference_image_path)
        
        # 读取参考影像获取元数据
        with rasterio.open(ref_img_path, **kwargs) as src:
            transform, crs, shape = src.transform, src.crs, (src.height, src.width)
        
        # 读取shp文件
        gdf = gpd.read_file(shp_path)

        # 进行转换
        label_image = self._convert_common(gdf, transform, shape, crs, column, fill_value, dtype)
        
        # 如果需要保存
        if output_path:
            metadata = self._save_image(label_image, Path(output_path), transform, crs, dtype, fill_value)
            return label_image, metadata
        return label_image

    

    def convert_custom(self,
                      shp_path: Union[str, Path],
                      bounds: Tuple[float, float, float, float],
                      resolution: Union[float, Tuple[float, float]],
                      crs: Union[str, rasterio.crs.CRS],
                      output_path: Optional[Union[str, Path]] = None,
                      column: Optional[str] = 'class',
                      fill_value: int = 0,
                      dtype: type = np.uint8) -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
        """
        自定义参数进行转换
        
        Parameters:
        -----------
        shp_path    shp文件路径
        bounds      图像边界 (minx, miny, maxx, maxy)
        resolution  分辨率，如果是单个值则x,y分辨率相同
        crs         坐标参考系统
        output_path 输出图像路径
        column      用于提取标签值的列名
        fill_value  背景填充值
        dtype       输出数组的数据类型
            
        Returns:
        --------
        label_image 转换后的标签图像
        metadata    如果output_path不为None，返回元数据字典
        """
        shp_path = Path(shp_path)
        
        minx, miny, maxx, maxy = bounds

        # 计算图像尺寸
        x_res, y_res = (resolution, resolution) if isinstance(resolution, (int, float)) else resolution
        width, height = int((maxx - minx) / x_res), int((maxy - miny) / y_res)
        
        # 创建仿射变换及输出信息
        transform = from_bounds(minx, miny, maxx, maxy, width, height)
        shape = (height, width)

        # 读取shp文件
        gdf = gpd.read_file(shp_path)

        # 进行转换
        label_image = self._convert_common(gdf, transform, shape, crs, column, fill_value, dtype)
        
        # 如果需要保存
        if output_path:
            metadata = self._save_image(label_image, Path(output_path), transform, crs, dtype, fill_value)
            return label_image, metadata
        return label_image

        
    def shp_bounds_in_target_crs(self, 
                                shapefile_path: str, 
                                target_crs: str,
                                padding: float = 0) -> Tuple[float, float, float, float]:

        """
        获取 shp 文件在目标 CRS 中的范围
        
        Parameters:
        -----------
        shapefile_path  shp 文件路径
        target_crs      目标 CRS
        padding         范围扩展量（与目标 CRS 单位相同）
        
        Returns:
        --------
        bounds          在目标 CRS 中的范围
        """
        gdf = gpd.read_file(shapefile_path)
        
        # 如果 CRS 不同，转换到目标 CRS
        if gdf.crs != target_crs:
            gdf = gdf.to_crs(target_crs)
            self.logger.info(f"将shp文件从源CRS({gdf.crs})转换到目标CRS({target_crs})...")
        
        bounds = gdf.total_bounds
        
        # 添加边界扩展
        if padding > 0:
            bounds = (
                bounds[0] - padding,
                bounds[1] - padding,
                bounds[2] + padding,
                bounds[3] + padding
            )
        
        return bounds

    def convert_bounds_to_target_crs(self,
                                   bounds: Tuple[float, float, float, float],
                                   source_crs: str,
                                   target_crs: str) -> Tuple[float, float, float, float]:
        """
        将范围从源CRS转换到目标CRS
        
        Parameters:
        -----------
        bounds          源CRS中的范围 (minx, miny, maxx, maxy)
        source_crs      源CRS
        target_crs      目标CRS
            
        Returns:
        --------
        target_bounds   在目标CRS中的范围
        """
        # 创建一个表示范围的几何图形
        bbox = box(*bounds)
        
        # 创建GeoDataFrame并转换CRS
        gdf_bbox = gpd.GeoDataFrame(geometry=[bbox], crs=source_crs)
        gdf_bbox_target = gdf_bbox.to_crs(target_crs)
        
        # 获取转换后的范围
        target_bounds = gdf_bbox_target.total_bounds
        
        self.logger.info(f"范围从 {source_crs} 转换到 {target_crs}:")
        self.logger.info(f"  源范围: {bounds}")
        self.logger.info(f"  目标范围: {target_bounds}")
        
        return target_bounds
    
    def visualize(self, 
                 image: np.ndarray, 
                 title: str = "Label Image",
                 cmap: str = 'tab20',
                 figsize: Tuple[int, int] = (10, 8)):
        """
        可视化转换结果
        
        Parameters:
        -----------
        image   标签图像
        title   图像标题
        cmap    颜色映射
        figsize 图像大小
        """
        plt.figure(figsize=figsize)
        plt.imshow(image, cmap=cmap)
        plt.colorbar(label='Class ID')
        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        plt.show()


    def visualize_png(self, 
                image: np.ndarray, 
                out_dir: str,
                title: str,
                rs_img_path: str = None,
                figsize: Tuple[int, int] = (20, 16),
                class_names: Optional[dict] = None,
                max_image_size: int = 3000):
        """
        可视化转换结果并保存到指定路径
        
        Parameters:
        -----------
        image       标签图像
        out_dir     输出文件夹路径
        title       图像标题
        rs_img      原始图像，用于叠加显示
        figsize     图像大小
        class_names 类别名称字典，格式为 {class_id: class_name}
        """
        # 创建输出文件夹（如果不存在）
        os.makedirs(out_dir, exist_ok=True)

        # 图像尺寸优化
        new_height, new_width = image.shape
        if max(image.shape) > max_image_size:
            scale = max_image_size / max(image.shape)
            new_height = int(image.shape[0] * scale)
            new_width = int(image.shape[1] * scale)
            image = np.array(Image.fromarray(image.astype(np.uint8)).resize((new_width, new_height), Image.NEAREST))
        dpi_width = new_width / figsize[0]
        dpi_height = new_height / figsize[1]
        dpi = min(dpi_width, dpi_height) 
            
        # 获取所有唯一的类别值
        unique_classes = np.unique(image)
        n_classes = len(unique_classes)
        
        # 创建自定义颜色映射
        colors = plt.colormaps['tab20'].resampled(n_classes)
        
        # 创建新的颜色映射列表，将class 0设置为透明灰色
        cmap_colors = []
        for i in range(n_classes):
            if unique_classes[i] == 0:
                cmap_colors.append((0.5, 0.5, 0.5, 0.1))  # 透明灰色
            else:
                cmap_colors.append(colors(i / n_classes))
        
        # 创建自定义颜色映射
        custom_cmap = ListedColormap(cmap_colors)
        
        # 创建图形
        fig, ax = plt.subplots(figsize=figsize)

        if rs_img_path is not None:
            with rasterio.open(rs_img_path) as rs_img:
                rs_img = rs_img.read(out_shape=(4, new_height, new_width), resampling=rasterio.enums.Resampling.nearest)
                rs_img = np.transpose(rs_img, (1, 2, 0))
                rs_img = np.array(Image.fromarray(rs_img.astype(np.uint8)))

                if len(rs_img.shape) == 3:  # RGB图像 or RGBA图像
                    ax.imshow(rs_img)
                else:  # 灰度图像
                    ax.imshow(rs_img, cmap='gray')
        
        # 显示图像
        im = ax.imshow(image, cmap=custom_cmap, vmin=unique_classes.min(), vmax=unique_classes.max(), interpolation='nearest')
        
        # 创建图例
        if class_names is None:
            class_names = {i:i for i in unique_classes}
        
        # 为每个类别创建图例
        legend_elements = []
        for class_id in unique_classes:
            color = custom_cmap((class_id - unique_classes.min()) / (unique_classes.max() - unique_classes.min()))
            label = class_names.get(class_id, f'Class {class_id}')
            legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=color, label=label))
        
        ax.legend(handles=legend_elements, 
                bbox_to_anchor=(1.01, 0.8, 0.15, 0.15), 
                loc='upper left',
                title="Classes",
                title_fontsize=16,
                fontsize=16,)
        
        # 设置标题和坐标轴
        plt.title(title, fontsize=20)
        plt.axis('off')
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图像
        output_file = os.path.join(out_dir, f"{title.replace(' ', '_')}.png")
        plt.savefig(output_file, bbox_inches='tight', dpi=dpi)
        self.logger.info(f'图像保存路径: {Path(output_file)}')
        plt.close()
        


if __name__ == "__main__":
    shp_path = './utils/shp/sample.shp'
    out_dir = './utils/out_reference'
    out_path = './utils/out_reference/sample.tif'

    # mode == 'reference'
    converter_ref = SHP2IMAGE(mode='reference', format='tif', log_level='INFO')
    ref_img_path = './utils/rs/sample.tif'
    label_img_ref = converter_ref.convert(
        shp_path=shp_path,
        reference_image_path=ref_img_path,
        output_path=out_path,
        column="class", 
        fill_value=0
    )
    converter_ref.visualize_png(label_img_ref[0], out_dir=out_dir, title='sample_Preview')
    converter_ref.visualize_png(label_img_ref[0], out_dir=out_dir, title='sample_PreviewBASE', rs_img_path=ref_img_path)



    # mode == 'custom'
    print()
    out_dir = './utils/out_custom'
    out_path = './utils/out_custom/sample.tif'
    converter_custom = SHP2IMAGE(mode='custom', format='tif', log_level='INFO')
    target_crs = 'EPSG:3857'
    label_img_custom = converter_custom.convert(
        shp_path=shp_path,
        output_path=out_path,
        bounds=converter_custom.shp_bounds_in_target_crs(shp_path, target_crs, padding=100),
        resolution=5, 
        crs = target_crs, 
        column="class",
        fill_value=0
    )
    converter_custom.visualize_png(label_img_custom[0], out_dir=out_dir, title='haikou_fast')
