from BatchLabel import BatchLabel

batch_handler = BatchLabel(overwrite=True, mode='all-in-one', format='tif', preview=False, preview_size=512)
rs_dir = './example1-tif'
shp_dir = './example1-shp'
out_dir = './example1-out'
batch_handler.run(shp_dir, rs_dir, out_dir)