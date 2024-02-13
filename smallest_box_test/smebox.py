import main as mn

#test if the find_obb function (hinting to 'oriented bounding box') is invariant under rotation

df = mn.load_mol2_file('1fm6.mol2')
df_rotation = mn.load_mol2_file('1fm6_rotated.mol2')
df_translation = mn.load_mol2_file('1fm6_translated.mol2')

points = mn.get_points(df)
rot_points = mn.get_points(df_rotation)
trans_points = mn.get_points(df_translation)

print(mn.find_obb(points))
