import sys
sys.path.append("..")
import main as mn

#location of the directory with the volsite discriptor directories
file_loc = sys.argv[1]
#csv file to add descriptors onto
descriptors = sys.argv[2]
if not descriptors.endswith('.csv'):
    raise Exception("The second input is not a csv file.")

for dir in file_loc:
    mn(dir, descriptors)







