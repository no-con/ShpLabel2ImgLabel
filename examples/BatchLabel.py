import os
import logging
from Shp2Img import SHP2IMAGE
from pathlib import Path
from typing import List, Optional, Tuple

class BatchLabel:
    def __init__(self, mode:str='for-each', overwrite:bool=True, preview:bool=True, preview_size=1024, format:str='tif', log_level:int=logging.INFO, shp2img_log_level:int=logging.WARNING):
        """
        Parameters:
        -----------
        mode : str
            模式：'for-each': 每幅遥感影像放在各自文件夹；'all-in-one': 所有结果放在同一个文件夹
            对于模式'for-each'，请确保每幅遥感影像的文件夹名与内部的遥感影像名、对应的shp文件夹名、内部的shp文件名相同
            对于模式'all-in-one'，请确保每幅遥感影像名与对应的shp文件名相同

        overwrite  : bool
            是否覆盖已有结果

        preview : bool
            是否输出低分辨率图片快速查看结果
        
        preview_size : int
            快速查看图片的分辨率

        format : str
            输出影像格式
        
        log_level : int
            日志级别
        """

        self.mode = mode
        if self.mode not in ['for-each', 'all-in-one']:
            raise ValueError("模式必须为 'for-each' 或 'all-in-one'")
        self.overwrite = overwrite
        self.preview = preview
        self.preview_size = preview_size

        if format not in ['tif', 'png', 'jpg', 'jpeg']:
            raise ValueError(f"不支持的图像格式: format={format}, 可选值: ['tif', 'png', 'jpg', 'jpeg']")
        self.format = format
        self.converter = SHP2IMAGE(mode='reference', format=format, log_level=shp2img_log_level)

        # 配置日志
        self.log_level = log_level
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        if not self.logger.handlers:
            # 创建处理器
            handler = logging.StreamHandler()
            handler.setLevel(log_level)
            
            # 创建格式器
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            
            # 添加处理器到logger
            self.logger.addHandler(handler)
            self.logger.setLevel(log_level)


    def _get_paths(self, file_name: str, shp_dir: Path, rs_dir: Path, out_dir: Path) -> Tuple[Path, Path, Path]:
        """
        根据模式获取单个文件的遥感影像、shp和输出路径。
        """
        rs_postfix = 'tif'
        shp_postfix = 'shp'
        label_img_postfix = 'tif'

        if self.mode == 'for-each':
            rs_path = rs_dir / file_name / f"{file_name}.{rs_postfix}"
            shp_path = shp_dir / file_name / f"{file_name}.{shp_postfix}"
            out_path = out_dir / file_name / f"{file_name}.{label_img_postfix}"
        else:  # 'all-in-one'
            rs_path = rs_dir / f"{file_name}.{rs_postfix}"
            shp_path = shp_dir / f"{file_name}.{shp_postfix}"
            out_path = out_dir / f"{file_name}.{label_img_postfix}"
        
        return rs_path, shp_path, out_path


    def run(self, Shp_dir, RS_dir, out_dir, class_field='class', to_process_list:list=None, class_names:dict=None):
        """
        开始处理
        
        Parameters:
        -----------
        Shp_dir: 标注shp文件夹
        RS_dir : 遥感影像文件夹
        out_dir: 输出结果文件夹
        class_field : shp文件中类别字段名，默认为'class'
        to_process_list : 需要处理的文件名列表, 默认为None, 即处理所有文件
        class_names : 类别名称字典，默认为None, 即使用默认类别名称
        """

        shp_dir = Path(Shp_dir)
        rs_dir = Path(RS_dir)
        out_dir = Path(out_dir)

        # 检查输入目录
        if not rs_dir.is_dir():
            self.logger.error(f"未找到遥感影像文件夹: {rs_dir}")
            return
        if not shp_dir.is_dir():
            self.logger.error(f"未找到标注shpfile文件夹: {shp_dir}")
            return


      # 创建输出目录
        out_dir.mkdir(parents=True, exist_ok=True)

        # 获取待处理文件列表
        if not to_process_list:
            # 根据模式确定如何获取文件名
            if self.mode == 'for-each':
                to_process_list = [d.name for d in rs_dir.iterdir() if d.is_dir()]
            elif self.mode == 'all-in-one': 
                to_process_list = [f.stem for f in rs_dir.glob('*.tif')]

        if not to_process_list:
            self.logger.warning("在遥感影像目录中未找到任何可处理的文件。")
            return


        self.logger.info(f"将处理 {len(to_process_list)} 个文件: {to_process_list}")
        self.logger.info('=========================== Process Start ===========================')

        for file_name in to_process_list:
            try:
                rs_path, shp_path, out_path = self._get_paths(file_name, shp_dir, rs_dir, out_dir)

                # 检查是否跳过
                if out_path.exists() and not self.overwrite:
                    self.logger.info(f"{file_name}: Skip! 输出文件已存在: {out_path}")
                    continue

                # 检查输入文件是否存在
                if not rs_path.exists():
                    self.logger.error(f"❌ {file_name}: Fail! 未找到遥感影像文件: {rs_path}")
                    continue
                if not shp_path.exists():
                    self.logger.error(f"❌ {file_name}: Fail! 未找到标注shpfile文件: {shp_path}")
                    continue
                
                # 确保输出目录存在 (对于'for-each'模式)
                out_path.parent.mkdir(parents=True, exist_ok=True)

                # 执行转换
                label_image, metadata = self.converter.convert(
                    shp_path=str(shp_path), 
                    reference_image_path=str(rs_path), 
                    output_path=str(out_path), 
                    column=class_field
                )

                # 生成快速预览
                if self.preview:
                    self.converter.visualize_png(
                        image=label_image, 
                        out_dir=str(out_path.parent), 
                        title=f"{file_name}_Preview",
                        class_names=class_names,
                        max_image_size=self.preview_size
                    )
                    self.converter.visualize_png(
                        image=label_image, 
                        out_dir=str(out_path.parent), 
                        title=f"{file_name}_PreviewBase", 
                        rs_img_path=str(rs_path),
                        class_names=class_names,
                        max_image_size=self.preview_size
                    )
                
                self.logger.info(f"✅ {file_name}: Success! size={metadata['height'], metadata['width']}")

            except Exception as e:
                # 捕获单个文件处理中的任何异常，防止整个程序中断
                self.logger.error(f"❌ {file_name}: 处理过程中发生未知错误: {e}", exc_info=True)
                continue
        
        self.logger.info('============================ Process End ============================\n')


if __name__ == '__main__':
    batch_handler = BatchLabel(overwrite=True, mode='all-in-one', format='tif')
    rs_dir = './utils/rs'
    shp_dir = './utils/shp'
    out_dir = './utils/out_batch/'
    batch_handler.run(shp_dir, rs_dir, out_dir)
