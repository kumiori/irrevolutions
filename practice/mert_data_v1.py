
import h5py
import numpy as np
import xml.etree.ElementTree as ET
xdmf_file = "./output/mert/traction-bar.xdmf"
tree = ET.parse(xdmf_file)
root = tree.getroot()

# Find the HDF5 data file path (assuming it's in a DataItem element)
data_item = root.find(".//DataItem")
hdf5_file_path = data_item.text.strip()
xdmf_file = "./output/mert/traction-bar.xdmf"
tree = ET.parse(xdmf_file)
root = tree.getroot()

# Find the HDF5 data file path (assuming it's in a DataItem element)
data_item = root.find(".//DataItem")
hdf5_file_path = data_item.text.strip()


# Open the HDF5 file
hdf5_file = h5py.File(hdf5_file_path, "r")

# You can list the available datasets in the HDF5 file
print("Available datasets in HDF5 file:")
for key in hdf5_file.keys():
    print(key)

# Access the dataset you want
dataset = hdf5_file["your_dataset_name"]



# Open the HDF5 file
hdf5_file = h5py.File(hdf5_file_path, "r")

# You can list the available datasets in the HDF5 file
print("Available datasets in HDF5 file:")
for key in hdf5_file.keys():
    print(key)

# Access the dataset you want
dataset = hdf5_file["your_dataset_name"]