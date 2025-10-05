from BatchLabel import BatchLabel

batch_handler = BatchLabel(overwrite=True, mode='for-each', format='tif', preview=True, preview_size=512)
rs_dir = './example2-tif'
shp_dir = './example2-shp'
out_dir = './example2-out'
batch_handler.run(shp_dir, rs_dir, out_dir)