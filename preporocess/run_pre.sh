python preprocess.py \
  --he ../data/data/CRC03/18459_LSP10375_US_SCAN_OR_001__092147-registered.ome.tif \
  --If ../data/data/CRC03/P37_S31_A24_C59kX_E15_20220106_014409_014236-zlib.ome.tiff \
  --out ../data/data/CRC03/preprocessed \
  --chunk-size 5000 > opt_output.log 2>&1&
