import logging
from BatchLabel import BatchLabel

batch_handler = BatchLabel(overwrite=True, mode='all-in-one', format='tif', preview=True, log_level=logging.INFO, shp2img_log_level=logging.INFO)
rs_dir = './example3-tif'
shp_dir = './example3-shp'
out_dir = './example3-out'
batch_handler.run(shp_dir, rs_dir, out_dir)