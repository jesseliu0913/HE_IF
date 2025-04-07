import tifffile

# Open the image
with tifffile.TiffFile('/playpen/jesse/HE_IF/data/data/CRC01/18459_LSP10353_US_SCAN_OR_001__093059-registered.ome.tif') as tif:
    # Get OME metadata and image shape
    ome_metadata = tif.ome_metadata
    series = tif.series[0]
    shape = series.shape
    dtype = series.dtype

print("Shape:", shape)
print("Data type:", dtype)
