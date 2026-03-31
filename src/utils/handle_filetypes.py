"""
Library to handle input and output filetypes.

Author: Louis Evans
Reviewer: Stefano Merlini
"""

import numpy as np
import pyvista as pv
import os
import tarfile
import io

def export_pvti(arr: np.ndarray, fname: str = None, extent_x = None, extent_y = None, extent_z = None):
    '''
    Export a 3d array as a pvti file format
        fname: str, file path and name to save under. A VTI pointed to by a PVTI file are saved in this location. If left blank, the name will default to:
                ./plasma_PVTI_DD_MM_YYYY_HR_MIN    
    '''

    if fname is None:
        import datetime as dt
        year = dt.datetime.now().year
        month = dt.datetime.now().month
        day = dt.datetime.now().day
        min = dt.datetime.now().minute
        hour = dt.datetime.now().hour

        fname = f'./plasma_PVTI_{day}_{month}_{year}_{hour}_{min}' #default fname to the current date and time 

    try: #check to ensure electron density has been added
        np.shape(arr)
        rnec = arr
    except:
        raise Exception('No electron density currently loaded!')

    # Create the spatial reference  
    grid = pv.ImageData()

    # Set the grid dimensions: shape + 1 because we want to inject our values on
    # the CELL data
    grid.dimensions = np.array(rnec.shape) + 1
    # Edit the spatial reference
    grid.origin = (0, 0, 0)  # The bottom left corner of the data set

    if extent_x is None:
        extent_x = (np.shape(arr)[0]) // 2
    if extent_y is None:
        extent_y = (np.shape(arr)[1]) // 2
    if extent_z is None:
        extent_z = (np.shape(arr)[2]) // 2

    xc, yc, zc = np.linspace(-extent_x, extent_x, arr.shape[0]), np.linspace(-extent_y, extent_y, arr.shape[1]), np.linspace(-extent_z, extent_z, arr.shape[2])

    #scaling
    x_size = np.max(xc) / ((np.shape(arr)[0])//2 )  #assuming centering about the origin
    y_size = np.max(yc) / ((np.shape(arr)[1])//2 ) 
    z_size = np.max(zc) / ((np.shape(arr)[2])//2 )
    grid.spacing = (x_size, y_size, z_size)  # These are the cell sizes along each axis

    # Add the data values to the cell data
    grid.cell_data["rnec"] = rnec.flatten(order="F")  # Flatten the array

    grid.save(f'{fname}.vti')

    print(f'VTI saved under {fname}.vti')

    #prep values to write the pvti, written to match the exported vti using pyvista
    relative_fname = fname.split('/')[-1]
    spacing_x = (2 * xc.max()) / np.shape(xc)[0]
    spacing_y = (2 * yc.max()) / np.shape(yc)[0]
    spacing_z = (2 * zc.max()) / np.shape(zc)[0]

    content = f'''<?xml version="1.0"?>
                    <VTKFile type="PImageData" version="0.1" byte_order="LittleEndian" header_type="UInt32" compressor="vtkZLibDataCompressor">
                    <PImageData WholeExtent="0 {np.shape(arr)[0]} 0 {np.shape(arr)[1]} 0 {np.shape(arr)[2]}" GhostLevel="0" Origin="0 0 0" Spacing="{x_size} {y_size} {z_size}">
                            <PCellData Scalars="rnec">
                                <PDataArray type="Float32" Name="rnec">
                                </PDataArray>
                            </PCellData>
                            <Piece Extent="0 {np.shape(arr)[0]} 0 {np.shape(arr)[1]} 0 {np.shape(arr)[2]}" Source="{relative_fname}.vti"/>
                    </PImageData>
                    </VTKFile>'''

    # write file
    with open(f'{fname}.pvti', 'w') as file:
        file.write(content)

    print(f'Scalar Domain electron density succesfully saved under {fname}.pvti !')

def pvti_readin(filename):
    '''
	Reads in data from pvti with filename, use this to read in electron number density data
	'''
    import vtk
    from vtk.util import numpy_support as vtk_np

    reader = vtk.vtkXMLPImageDataReader()
    reader.SetFileName(filename)
    reader.Update()

    data = reader.GetOutput()
    dim = data.GetDimensions()
    spacing = np.array(data.GetSpacing())

    v = vtk_np.vtk_to_numpy(data.GetCellData().GetArray(0))
    n_comp = data.GetCellData().GetArray(0).GetNumberOfComponents()

    vec = [int(i-1) for i in dim]

    if(n_comp > 1):
        vec.append(n_comp)

    if(n_comp > 2):
        img = v.reshape(vec,order="F")[0:dim[0]-1,0:dim[1]-1,0:dim[2]-1,:]
    else:
        img = v.reshape(vec,order="F")[0:dim[0]-1,0:dim[1]-1,0:dim[2]-1]

    #dim = img.shape
    return img, img.shape, spacing

def hdf_readin(filename, *, prefer_ye=True, force_override=True, verbose=False,
               return_tele=False, return_z=False):
    """
    Load a FLASH/yt-readable HDF5 file and return an electron-density field on a
    uniform covering grid.

    What this does
    -------------
    1) Loads the dataset with yt.
    2) Defines a derived fluid field ("flash","ne") for electron number density.
       - If a field ("flash","ye") exists and prefer_ye=True:
            ne = N_A * rho * Ye
         (This matches your original scaling: Ye = Z/A, N_A = 6.022e23)
       - Otherwise falls back to a simple scaled proxy:
            ne = 5e23 * rho
    3) Optionally defines a derived field ("flash","z") for the mean ionic charge
       state, computed from the FLASH fields ye (= Z/A) and sumy (= 1/A):
            z = ye / sumy
    4) Builds a covering_grid at the maximum AMR level and returns:
       - ne: yt array on the covering grid
       - dims: integer grid dimensions
       - spacing: cell spacing per axis (yt quantities)
       - extras (only when return_tele or return_z is True): dict containing
         any of the keys "tele" and/or "z" as yt arrays on the covering grid.

    Parameters
    ----------
    filename : str or Path
        Path to the HDF5 file.
    prefer_ye : bool
        If True, use ("flash","ye") when available to compute ne.
    force_override : bool
        If True, overwrite derived fields if they already exist in the session.
    verbose : bool
        If True, print field availability and grid info.
    return_tele : bool
        If True, also extract the electron temperature field ("flash","tele")
        and include it in the returned extras dict.
    return_z : bool
        If True, derive the mean ionic charge state z = ye / sumy and include
        it in the returned extras dict.  Requires ("flash","ye") and
        ("flash","sumy") to be present in the file.

    Returns
    -------
    ne : yt.units.yt_array.YTArray
        Electron density field on covering grid.
    dims : ndarray (int)
        Grid dimensions at max AMR level.
    spacing : list
        Cell spacing along each axis (yt quantities).
    extras : dict, only present when return_tele=True or return_z=True
        Optional additional fields on the covering grid:
        - "tele": electron temperature (if return_tele=True)
        - "z": mean ionic charge state (if return_z=True)
    """
    import yt

    ds = yt.load(filename)

    # ---- Detect field availability
    field_ye = ("flash", "ye")
    field_sumy = ("flash", "sumy")
    field_tele = ("flash", "tele")
    has_ye = (field_ye in ds.field_list) or (field_ye in ds.derived_field_list)
    has_sumy = (field_sumy in ds.field_list) or (field_sumy in ds.derived_field_list)
    has_tele = (field_tele in ds.field_list) or (field_tele in ds.derived_field_list)

    if verbose:
        print(f"[hdf_readin] Loaded: {filename}")
        print(f"[hdf_readin] has_ye={has_ye} (prefer_ye={prefer_ye})")
        print(f"[hdf_readin] has_sumy={has_sumy}")
        print(f"[hdf_readin] has_tele={has_tele}")

    # ---- Derived field: ne
    # Keep your original coefficient (Avogadro's number in 1/mol)
    N_A = 6.022e23

    # If you want yt-consistent number-density units, consider:
    # ne_units = "1/cm**3" or "1/m**3" (requires rho and Ye units to be consistent)
    # For now we keep your original unit expression to avoid breaking downstream code.
    ne_units = "code_mass/code_length**3"

    use_ye = bool(prefer_ye and has_ye)

    def _ne(field, data):
        rho = data[("flash", "dens")]
        if use_ye:
            ye = data[field_ye]
            return N_A * rho * ye
        else:
            return 5e23 * rho

    ds.add_field(
        name=("flash", "ne"),
        function=_ne,
        sampling_type="local",
        units=ne_units,
        force_override=force_override,
    )

    # ---- Derived field: z = ye / sumy  (Z/A divided by 1/A = Z)
    if return_z:
        if not (has_ye and has_sumy):
            raise ValueError(
                "[hdf_readin] return_z=True requires both ('flash','ye') and "
                "('flash','sumy') to be present in the file, but one or both "
                "are missing."
            )

        def _z(field, data):
            return data[field_ye] / data[field_sumy]

        ds.add_field(
            name=("flash", "z"),
            function=_z,
            sampling_type="local",
            units="dimensionless",
            force_override=force_override,
        )

    # ---- Build covering grid at max AMR level
    level = ds.index.max_level
    dims = ds.domain_dimensions * (ds.refine_by ** level)

    cube = ds.covering_grid(
        level=level,
        left_edge=ds.domain_left_edge,
        dims=dims,
    )

    ne = cube[("flash", "ne")]

    # ---- Cell spacing (as yt quantities)
    spacing = [
        (ds.domain_right_edge[i] - ds.domain_left_edge[i]) / dims[i]
        for i in range(len(dims))
    ]

    if verbose:
        print(f"[hdf_readin] max_level={level}")
        print(f"[hdf_readin] dims={tuple(int(x) for x in np.array(dims))}")
        print(f"[hdf_readin] spacing={spacing}")

    # ---- Collect optional extra fields
    if return_tele or return_z:
        if return_tele and not has_tele:
            raise ValueError(
                "[hdf_readin] return_tele=True requires ('flash','tele') to be "
                "present in the file, but it is missing."
            )
        extras = {}
        if return_tele:
            extras["tele"] = cube[("flash", "tele")]
        if return_z:
            extras["z"] = cube[("flash", "z")]
        return ne, dims, spacing, extras

    return ne, dims, spacing

def hdf_to_pvti(hdf_filename, pvti_filename):
    '''
    convert hdf5 format to pvti format
    '''

    ne, dims, spacing = hdf_readin(hdf_filename)
    extent_x = (dims[0]*spacing[0])/2
    extent_y = (dims[1]*spacing[1])/2
    extent_z = (dims[2]*spacing[2])/2
    
    export_pvti(ne, fname = pvti_filename, extent_x = extent_x, extent_y = extent_y, extent_z = extent_z)

import h5py

def save_jax_matrix_to_hdf5(input_matrix, *, filename = None, filepath = None, dataset_name = 'data', compression = 'gzip', compression_level = 4):
    """
    Compress a JAX matrix and save it into an HDF5 file.

    :param input_matrix: The JAX array to be saved.
    :type input_matrix: jax.numpy.DeviceArray (REQUIRED)

    :param filename: What to call the resultant file.
    :type filename: str (default: None, sets filename to "ray_output" + current date-time stamp)

    :param filepath: Path to save the created HDF5 file too.
    :type filepath: str (default: None, saves to the current working directory)

    :param dataset_name: Name of the dataset inside the HDF5 file.
    :type dataset_name: str (default: 'data')

    :param compression: Compression algorithm to use (e.g., 'gzip', 'lzf', or None).
    :type compression: str or None (default: "gzip")

    :param compression_level: Compression level for gzip (1-9). Only used if compression is 'gzip'.
    :type compression_level: int (default: 4)

    :return: Returns the filepath the hdf5 saved matrix was written too.
    :rtype: str
    """

    # Convert JAX array to NumPy array for saving
    numpy_array = np.asarray(input_matrix, dtype = np.float32).tolist()

    if filename is None:
        from datetime import datetime
        filename = "ray_output" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".hdf5"

    if filepath is None:
        filepath = os.getcwd()

    filepath = os.path.join(filepath, filename)
    with h5py.File(filepath, 'w') as h5file:
        h5file.create_dataset(
            dataset_name,
            data = numpy_array,
            compression = compression,
            compression_opts = compression_level
        )

    print(f"Matrix saved to '{filepath}' with compression='{compression}' at compression_level='{compression_level}'.")
    return filepath, filename

def compress_matrix_to_hdf5_BytesIO(input_matrix, *, dataset_name = 'data', compression = 'gzip', compression_level = 4):
    """
    Compresses a JAX matrix into an in-memory HDF5 file (as bytes).

    :param input_matrix: The JAX array to be converted.
    :type input_matrix: jax.numpy.DeviceArray (REQUIRED)

    :param filename: What to call the resultant file.
    :type filename: str (default: None, sets filename to "ray_output" + current date-time stamp)

    :param filepath: Path to save the created HDF5 file too.
    :type filepath: str (default: None, saves to the current working directory)

    :param dataset_name: Name of the dataset inside the HDF5 file.
    :type dataset_name: str (default: 'data')

    :param compression: Compression algorithm to use (e.g., 'gzip', 'lzf', or None).
    :type compression: str or None (default: "gzip")

    :param compression_level: Compression level for gzip (1-9). Only used if compression is 'gzip'.
    :type compression_level: int (default: 4)

    :return: Returns the filename and data as a Byte Stream
    :rtype: Tuple->(str, BytesIO buffer)
    """

    # Convert JAX array to NumPy array for saving
    numpy_array = np.asarray(input_matrix, dtype = np.float32).tolist()

    # Setup a byte buffer for hdf5 to write into in memory
    hdf5_buffer = io.BytesIO()

    with h5py.File(hdf5_buffer, 'w') as h5file:
        h5file.create_dataset(
            dataset_name,
            data = numpy_array,
            compression = compression,
            compression_opts = compression_level
        )

    hdf5_buffer.seek(0) # Reset pointer to start
    return hdf5_buffer.getvalue() # return the binary data

def move_file_to_tar_gz(tar_gz_path, filepath, arcname = None):
    mode = 'a:gz' if os.path.exists(tar_gz_path) else 'w:gz'

    try:
        with tarfile.open(tar_gz_path, mode) as tar:
            tar.add(file_path, arcname = arcname)
        os.remove(file_path)  # Only remove if no exception occurred during add
    except Exception as e:
        print(f"Failed to add {file_path} to archive: {e}")

def stream_data_to_tar_gz(tar_gz_path, filename, content_bytes):
    mode = 'a:gz' if os.path.exists(tar_gz_path) else 'w:gz'
    print(tar_gz_path)
        
    try:
        with tarfile.open(tar_gz_path, mode) as tar:
            # raw content to write
            fileobj = io.BytesIO(content_bytes)

            # metadata
            info = tarfile.TarInfo(name = filename)
            info.size = len(content_bytes)

            # stream data to tar.gz file
            tar.addfile(info, fileobj)
    except Exception as e:
        print(f"Failed to stream data to archive: {e}")

def load_hdf5_from_tar_gz(tar_gz_path, member_name):
    """
    Load an HDF5 file from inside a tar.gz archive directly into memory.

    :return: h5py.File object (in read-only mode)
    """
    with tarfile.open(tar_gz_path, 'r:gz') as tar:
        try:
            file_bytes = tar.extractfile(tar.getmember(member_name)).read()
            hdf5_file = h5py.File(io.BytesIO(file_bytes), 'r')
            return hdf5_file
        except KeyError:
            print(f"File '{member_name}' not found in archive.")
            return None

def load_hdf5_from_tar_gz(tar_gz_path, member_name):
    """
    Load an HDF5 file from inside a tar.gz archive directly into memory.

    :return: h5py.File object (in read-only mode)
    """

    with tarfile.open(tar_gz_path, 'r:gz') as tar:
        try:
            file_bytes = tar.extractfile(tar.getmember(member_name)).read()
            return h5py.File(io.BytesIO(file_bytes), 'r')
        except KeyError:
            print(f"\nFile '{member_name}' not found in archive.")
            return None

def load_array_member_from_hdf5_tar_gz(tar_gz_path, member_name):
    h5file = load_hdf5_from_tar_gz(tar_gz_path, member_name)

    if h5file is not None:
        data = h5file['data'][:] # Read dataset into NumPy
        h5file.close()

        print("\nLoaded member shape & type: ", data.shape, data.dtype)

        return data
    else:
        print(f"\nCan't load data from member, '{member_name}'.")
        return None