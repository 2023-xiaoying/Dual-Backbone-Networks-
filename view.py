from helper_ply import read_ply, write_ply
import numpy as np

# path = '/home/chenke/桌面/DATA1/port/grid_0.500/area_1.ply'
# data = read_ply(path)
# mask = data['class'] == 1
# print(data['class'][mask].shape)
path1 = '/media/hello/76FCB52FFCB4EB0F/data/Toronto/original_ply/L002.ply'
path2 = '/media/hello/76FCB52FFCB4EB0F/code/FPWS_Net/test/Log_2024-01-11_09-07-55/val_preds/L002.ply'
pc_file = '/media/hello/76FCB52FFCB4EB0F/L002_pred.ply'
data1 = read_ply(path1)
data2 = read_ply(path2)
xyz = np.vstack((data1['x'], data1['y'], data1['z'])).T
rgb = np.vstack((data1['red'], data1['green'], data1['blue'])).T
label = np.vstack((data1['class']))
invalid_idx = np.where(label == 0)[0]
label = np.delete(label, invalid_idx, axis=0)
xyz = np.delete(xyz, invalid_idx, axis=0)
rgb = np.delete(rgb, invalid_idx, axis=0)
pred = np.vstack((data2['pred']))
list = pred == label
new = list + 0
new = np.array(new, dtype=np.uint8)
# dtype=np.uint8
write_ply(pc_file, [xyz, rgb, label, pred,new], ['x', 'y', 'z', 'red', 'green', 'blue', 'class', 'pred','diff'])
