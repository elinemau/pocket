import sys
sys.path.append("..")
import main as mn

#test if the find_obb function (hinting to 'oriented bounding box') is invariant under rotation

df = mn.load_mol2_file('1fm6.mol2')
df_rotation = mn.load_mol2_file('1fm6_rot4.mol2')
df_translation = mn.load_mol2_file('1fm6_trans.mol2')

points = mn.get_points(df)
rot_points = mn.get_points(df_rotation)
trans_points = mn.get_points(df_translation)

print(mn.find_obb(points))
print(mn.find_obb(rot_points))
print(mn.find_obb(trans_points))
