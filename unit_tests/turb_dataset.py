import image_slicer

tiles = image_slicer.slice('/home/albert/PycharmProjects/testConda/images/10006.jpg', 1024, save=False)
image_slicer.save_tiles(tiles, directory='/home/albert/PycharmProjects/testConda/images/10006', prefix='slice', format='png')


