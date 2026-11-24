import h5py as hf

fname = "/home/sac/data_67/Nowcasting/data/required_valid_south_test/202207010000.HDF5"
with hf.File(fname, 'r') as f:
    keys = f.keys() # Call the keys() method
    print(type(keys))
    for key in keys:
        print(f[key].shape)