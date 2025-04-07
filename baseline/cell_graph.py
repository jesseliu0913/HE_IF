import os
import numpy as np
import pandas as pd
import json
import pickle
import matplotlib
import matplotlib.path as mplPath
import matplotlib.pyplot as plt
import networkx as nx
import warnings
from scipy.spatial import Delaunay
from tqdm.notebook import tqdm  

RADIUS_RELAXATION = 0.1
NEIGHBOR_EDGE_CUTOFF = 20.76
UM_PER_PIXEL = 0.3775

def plot_voronoi_polygons(voronoi_polygons, voronoi_polygon_colors=None):
    if isinstance(voronoi_polygons, nx.Graph):
        voronoi_polygons = [voronoi_polygons.nodes[n]['voronoi_polygon'] for n in voronoi_polygons.nodes]

    if voronoi_polygon_colors is None:
        voronoi_polygon_colors = ['w'] * len(voronoi_polygons)
    assert len(voronoi_polygon_colors) == len(voronoi_polygons)

    xmax = 0
    ymax = 0
    for polygon, polygon_color in zip(voronoi_polygons, voronoi_polygon_colors):
        x, y = polygon[:, 0], polygon[:, 1]
        plt.fill(x, y, facecolor=polygon_color, edgecolor='k', linewidth=0.5)
        xmax = max(xmax, x.max())
        ymax = max(ymax, y.max())

    plt.xlim(0, xmax)
    plt.ylim(0, ymax)
    return

def plot_graph(G, node_colors=None):
    node_coords = [G.nodes[n]['center_coord'] for n in G.nodes]
    node_coords = np.stack(node_coords, 0)

    if node_colors is None:
        unique_cell_types = sorted(set([G.nodes[n]['cell_type'] for n in G.nodes]))
        cell_type_to_color = {ct: matplotlib.cm.get_cmap("tab20")(i % 20) for i, ct in enumerate(unique_cell_types)}
        node_colors = [cell_type_to_color[G.nodes[n]['cell_type']] for n in G.nodes]
    assert len(node_colors) == node_coords.shape[0]

    for (i, j, edge_type) in G.edges.data():
        xi, yi = G.nodes[i]['center_coord']
        xj, yj = G.nodes[j]['center_coord']
        if edge_type['edge_type'] == 'neighbor':
            plotting_kwargs = {"c": "k",
                               "linewidth": 1,
                               "linestyle": '-'}
        else:
            plotting_kwargs = {"c": (0.4, 0.4, 0.4, 1.0),
                               "linewidth": 0.3,
                               "linestyle": '--'}
        plt.plot([xi, xj], [yi, yj], zorder=1, **plotting_kwargs)

    plt.scatter(node_coords[:, 0],
                node_coords[:, 1],
                s=10,
                c=node_colors,
                linewidths=0.3,
                zorder=2)
    plt.xlim(0, node_coords[:, 0].max() * 1.01)
    plt.ylim(0, node_coords[:, 1].max() * 1.01)
    return

def calcualte_voronoi_from_coords(x, y, xmax=None, ymax=None):
    from geovoronoi import voronoi_regions_from_coords
    from shapely import geometry
    xmax = 1.01 * max(x) if xmax is None else xmax
    ymax = 1.01 * max(y) if ymax is None else ymax
    boundary = geometry.Polygon([[0, 0], [xmax, 0], [xmax, ymax], [0, ymax]])
    coords = np.stack([
        np.array(x).reshape((-1,)),
        np.array(y).reshape((-1,))], 1)
    region_polys, _ = voronoi_regions_from_coords(coords, boundary)
    voronoi_polygons = [np.array(list(region_polys[k].exterior.coords)) for k in region_polys]
    return voronoi_polygons

def build_graph_from_cell_coords(cell_data, voronoi_polygons):
    save_polygon = True
    if not len(cell_data) == len(voronoi_polygons):
        warnings.warn("Number of cells does not match number of voronoi polygons")
        save_polygon = False

    coord_ar = np.array(cell_data[['CellID', 'X', 'Y']])
    G = nx.Graph()
    node_to_cell_mapping = {}
    for i, row in enumerate(coord_ar):
        vp = voronoi_polygons[i] if save_polygon else None
        G.add_node(i, voronoi_polygon=vp)
        node_to_cell_mapping[i] = row[0]

    print("Computing Delaunay triangulation...")
    dln = Delaunay(coord_ar[:, 1:3])
    neighbors = [set() for _ in range(len(coord_ar))]
    
    print("Adding edges to graph...")
    for t in tqdm(dln.simplices, desc="Processing triangles"):
        for v in t:
            neighbors[v].update(t)
    
    for i, ns in tqdm(enumerate(neighbors), desc="Creating edges", total=len(neighbors)):
        for n in ns:
            if i != n:  
                G.add_edge(int(i), int(n))

    return G, node_to_cell_mapping

def build_graph_from_voronoi_polygons(voronoi_polygons, radius_relaxation=RADIUS_RELAXATION):
    G = nx.Graph()

    polygon_vertices = []
    vertice_identities = []
    for i, polygon in enumerate(voronoi_polygons):
        G.add_node(i, voronoi_polygon=polygon)
        polygon_vertices.append(polygon)
        vertice_identities.append(np.ones((polygon.shape[0],)) * i)

    polygon_vertices = np.concatenate(polygon_vertices, 0)
    vertice_identities = np.concatenate(vertice_identities, 0).astype(int)
    
    for i, polygon in tqdm(enumerate(voronoi_polygons), desc="Building graph from Voronoi", total=len(voronoi_polygons)):
        path = mplPath.Path(polygon)
        points_inside = np.where(path.contains_points(polygon_vertices, radius=radius_relaxation) +
                                 path.contains_points(polygon_vertices, radius=-radius_relaxation))[0]
        id_inside = set(vertice_identities[points_inside])
        for j in id_inside:
            if j > i:
                G.add_edge(int(i), int(j))
    return G

def build_voronoi_polygon_to_cell_mapping(G, voronoi_polygons, cell_data):
    cell_coords = np.array(list(zip(cell_data['X'], cell_data['Y']))).reshape((-1, 2))
    cells_in_polygon = {}
    
    for i, polygon in tqdm(enumerate(voronoi_polygons), desc="Mapping cells to polygons", total=len(voronoi_polygons)):
        path = mplPath.Path(polygon)
        _cell_ids = cell_data.iloc[np.where(path.contains_points(cell_coords))[0]]
        _cells = list(_cell_ids[['CellID', 'X', 'Y']].values)
        cells_in_polygon[i] = _cells

    def get_point_reflection(c1, c2, c3):
        x1, y1 = c1
        x2, y2 = c2
        x3, y3 = c3
        if x2 == x3:
            return (2 * x2 - x1, y1)
        m = (y3 - y2) / (x3 - x2)
        c = (x3 * y2 - x2 * y3) / (x3 - x2)
        d = (float(x1) + (float(y1) - c) * m) / (1 + m**2)
        x4 = 2 * d - x1
        y4 = 2 * d * m - y1 + 2 * c
        return (x4, y4)

    voronoi_polygon_to_cell_mapping = {}
    for i, polygon in tqdm(enumerate(voronoi_polygons), desc="Creating 1-to-1 polygon-cell mapping", total=len(voronoi_polygons)):
        path = mplPath.Path(polygon)
        if len(cells_in_polygon[i]) == 1:
            voronoi_polygon_to_cell_mapping[i] = cells_in_polygon[i][0][0]
        elif len(cells_in_polygon[i]) == 0:
            continue
        else:
            polygon_edges = [(polygon[_i], polygon[_i + 1]) for _i in range(-1, len(polygon) - 1)]
            neighbor_cells = sum([cells_in_polygon[j] for j in G.neighbors(i)], [])
            reflection_points = np.concatenate(
                [[get_point_reflection(cell[1:], edge[0], edge[1]) for edge in polygon_edges]
                    for cell in neighbor_cells], 0)
            reflection_points = reflection_points[np.where(path.contains_points(reflection_points))]
            dists = [((reflection_points - c[1:])**2).sum(1).min(0) for c in cells_in_polygon[i]]
            if not np.min(dists) < 0.01:
                warnings.warn("Cannot find the exact center cell for polygon %d" % i)
            voronoi_polygon_to_cell_mapping[i] = cells_in_polygon[i][np.argmin(dists)][0]
    return voronoi_polygon_to_cell_mapping

def assign_attributes(G,
                      cell_data,
                      node_to_cell_mapping,
                      edge_kwargs={
                          'neighbor_edge_cutoff': NEIGHBOR_EDGE_CUTOFF,
                          'um_per_pixel': UM_PER_PIXEL}):
    assert set(G.nodes) == set(node_to_cell_mapping.keys())
    biomarkers = sorted([c for c in cell_data.columns if c.startswith('BM-')])

    additional_features = sorted([
        c for c in cell_data.columns if c not in biomarkers + ['CellID', 'X', 'Y', 'CELL_TYPE']])

    cell_to_node_mapping = {v: k for k, v in node_to_cell_mapping.items()}
    node_properties = {}
    
    for _, cell_row in tqdm(cell_data.iterrows(), desc="Assigning node attributes", total=len(cell_data)):
        cell_id = cell_row['CellID']
        if cell_id not in cell_to_node_mapping:
            continue
        node_index = cell_to_node_mapping[cell_id]
        p = {"cell_id": cell_id}
        p["center_coord"] = (cell_row['X'], cell_row['Y'])
        if "CELL_TYPE" in cell_row:
            p["cell_type"] = cell_row["CELL_TYPE"]
        else:
            p["cell_type"] = "Unassigned"
        biomarker_expression_dict = {bm.split('BM-')[1]: cell_row[bm] for bm in biomarkers}
        p["biomarker_expression"] = biomarker_expression_dict
        for feat_name in additional_features:
            p[feat_name] = cell_row[feat_name]
        node_properties[node_index] = p

    nx.set_node_attributes(G, node_properties)
    
    print("Calculating edge types...")
    edge_properties = get_edge_type(G, **edge_kwargs)
    nx.set_edge_attributes(G, edge_properties)
    return G

def get_edge_type(G, neighbor_edge_cutoff=NEIGHBOR_EDGE_CUTOFF, um_per_pixel=UM_PER_PIXEL):
    edge_properties = {}
    for (i, j) in tqdm(G.edges, desc="Computing edge properties", total=G.number_of_edges()):
        ci = G.nodes[i]['center_coord']
        cj = G.nodes[j]['center_coord']
        dist = np.linalg.norm(np.array(ci) - np.array(cj), ord=2)
        edge_properties[(i, j)] = {
            "distance": dist,
            "edge_type": "neighbor" if dist < neighbor_edge_cutoff/um_per_pixel else "distant"
        }
    return edge_properties

def process_dataframe_for_graph(df, 
                               x_col='X_centroid', 
                               y_col='Y_centroid',
                               cell_id_col='CellID',
                               cell_type_col='CELL_TYPE',
                               biomarker_prefix=['BM_', 'marker_'],
                               feature_cols=None):
    """
    Process a DataFrame directly for graph construction without saving to CSV
    
    Args:
        df (pd.DataFrame): DataFrame containing cell data
        x_col (str): Column name for X coordinates
        y_col (str): Column name for Y coordinates
        cell_id_col (str): Column name for cell IDs
        cell_type_col (str): Column name for cell types
        biomarker_prefix (list): Prefixes used to identify biomarker columns
        feature_cols (list): Columns to use as additional features
        
    Returns:
        tuple: (cell_coords_df, cell_types_df, cell_biomarkers_df, cell_features_df)
    """
    print("Preparing data for graph construction...")
    df_copy = df.copy()
    
    if cell_id_col not in df_copy.columns:
        df_copy[cell_id_col] = df_copy.index
    
    original_cols = df_copy.columns.tolist()

    col_mapping = {col: col.upper() for col in original_cols}
    df_copy.columns = [col.upper() for col in df_copy.columns]
    
    cell_id_col_upper = cell_id_col.upper()
    x_col_upper = x_col.upper()
    y_col_upper = y_col.upper()
    if cell_type_col:
        cell_type_col_upper = cell_type_col.upper()
    
    coords_df = df_copy[[cell_id_col_upper, x_col_upper, y_col_upper]].copy()
    coords_df.columns = ['CellID', 'X', 'Y'] 
    
    if cell_type_col and cell_type_col_upper in df_copy.columns:
        cell_types_df = df_copy[[cell_id_col_upper, cell_type_col_upper]].copy()
        cell_types_df.columns = ['CellID', 'CELL_TYPE']
    else:
        cell_types_df = pd.DataFrame({'CellID': df_copy[cell_id_col_upper], 'CELL_TYPE': 'Unknown'})
    
    biomarker_cols = []
    for prefix in biomarker_prefix:
        prefix = prefix.upper()
        biomarker_cols.extend([col for col in df_copy.columns if col.startswith(prefix)])
    
    if biomarker_cols:
        biomarker_cols_with_id = [cell_id_col_upper] + biomarker_cols
        biomarker_df = df_copy[biomarker_cols_with_id].copy()
        
        renamed_cols = {col: f'BM-{col.replace("BM_", "").replace("MARKER_", "")}' 
                       for col in biomarker_cols}
        renamed_cols[cell_id_col_upper] = 'CellID'
        
        biomarker_df.rename(columns=renamed_cols, inplace=True)
    else:
        biomarker_df = None
    
    if feature_cols is None:
        feature_cols = ['AREA', 'MAJORAXISLENGTH', 'MINORAXISLENGTH', 'ECCENTRICITY', 
                       'SOLIDITY', 'EXTENT', 'ORIENTATION']
    
    feature_cols_upper = [col.upper() for col in feature_cols]
    available_features = [col for col in feature_cols_upper if col in df_copy.columns]
    
    if available_features:
        features_with_id = [cell_id_col_upper] + available_features
        features_df = df_copy[features_with_id].copy()
        
        col_renaming = {cell_id_col_upper: 'CellID'}
        features_df.rename(columns=col_renaming, inplace=True)
    else:
        features_df = None
    
    return coords_df, cell_types_df, biomarker_df, features_df

def construct_graph_from_dataframe(df,
                                 region_id='region',
                                 x_col='X_centroid',
                                 y_col='Y_centroid',
                                 cell_id_col='CellID',
                                 cell_type_col='CELL_TYPE',
                                 biomarker_prefix=['BM_', 'marker_'],
                                 feature_cols=None,
                                 edge_kwargs={
                                     'neighbor_edge_cutoff': NEIGHBOR_EDGE_CUTOFF,
                                     'um_per_pixel': UM_PER_PIXEL},
                                 graph_source='polygon',
                                 voronoi_polygon_img_output=None,
                                 graph_img_output=None,
                                 figsize=10):
    print(f"Constructing graph for region: {region_id}")
    
    coords_df, types_df, biomarker_df, features_df = process_dataframe_for_graph(
        df, x_col, y_col, cell_id_col, cell_type_col, biomarker_prefix, feature_cols
    )
    
    cell_data = coords_df
    
    print("Calculating Voronoi polygons...")
    voronoi_polygons = calcualte_voronoi_from_coords(cell_data['X'], cell_data['Y'])
    
    if types_df is not None:
        print("Merging cell type data...")
        if set(types_df['CellID']) != set(cell_data['CellID']):
            warnings.warn("Cell IDs in cell types do not match coordinates")
        shared_ids = set(types_df['CellID']).intersection(set(cell_data['CellID']))
        cell_data = cell_data[cell_data['CellID'].isin(shared_ids)]
        cell_data = cell_data.merge(types_df, on='CellID')
    
    if biomarker_df is not None:
        print("Merging biomarker data...")
        if set(biomarker_df['CellID']) != set(cell_data['CellID']):
            warnings.warn("Cell IDs in biomarker data do not match coordinates")
        shared_ids = set(biomarker_df['CellID']).intersection(set(cell_data['CellID']))
        cell_data = cell_data[cell_data['CellID'].isin(shared_ids)]
        cell_data = cell_data.merge(biomarker_df, on='CellID')
    
    if features_df is not None:
        print("Merging feature data...")
        if set(features_df['CellID']) != set(cell_data['CellID']):
            warnings.warn("Cell IDs in feature data do not match coordinates")
        shared_ids = set(features_df['CellID']).intersection(set(cell_data['CellID']))
        cell_data = cell_data[cell_data['CellID'].isin(shared_ids)]
        cell_data = cell_data.merge(features_df, on='CellID')
    
    if graph_source == 'polygon':
        print("Building graph from Voronoi polygons...")
        G = build_graph_from_voronoi_polygons(voronoi_polygons)
        node_to_cell_mapping = build_voronoi_polygon_to_cell_mapping(G, voronoi_polygons, cell_data)
        G = G.subgraph(node_to_cell_mapping.keys())
    elif graph_source == 'cell':
        print("Building graph from cell coordinates...")
        G, node_to_cell_mapping = build_graph_from_cell_coords(cell_data, voronoi_polygons)
    else:
        raise ValueError("graph_source must be either 'polygon' or 'cell'")
    
    print("Assigning node and edge attributes...")
    G = assign_attributes(G, cell_data, node_to_cell_mapping, edge_kwargs=edge_kwargs)
    G.region_id = region_id
    
    if voronoi_polygon_img_output is not None:
        print(f"Saving Voronoi polygon visualization to {voronoi_polygon_img_output}")
        plt.clf()
        plt.figure(figsize=(figsize, figsize))
        plot_voronoi_polygons(G)
        plt.axis('scaled')
        plt.savefig(voronoi_polygon_img_output, dpi=300, bbox_inches='tight')
    
    if graph_img_output is not None:
        print(f"Saving graph visualization to {graph_img_output}")
        plt.clf()
        plt.figure(figsize=(figsize, figsize))
        plot_graph(G)
        plt.axis('scaled')
        plt.savefig(graph_img_output, dpi=300, bbox_inches='tight')
    
    print(f"Graph construction complete: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G