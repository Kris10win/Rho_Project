import pandas as pd
import numpy as np
import gffutils
from Bio import SeqIO
from BCBio import GFF

# --------------------------------------------------------------------------------
# File paths
# --------------------------------------------------------------------------------
forward_file = '211_212_213_1328_40_R1_M14_M30_vs_234_235_236_1325_BCM_m14_M30_forward_strand_differential_table_window.txt'
reverse_file = '211_212_213_1328_40_R1_M14_M30_vs_234_235_236_1325_BCM_m14_M30_reverse_strand_differential_table_window.txt'
gff_file = 'NC_000913.2.gff3'
db_file = 'NC_000913.2.db'
tss_file_path = 'TSS.csv'
tss_df = pd.read_csv(tss_file_path)

# --------------------------------------------------------------------------------
# Create or load the GFF3 database
# --------------------------------------------------------------------------------
try:
    db = gffutils.FeatureDB(db_file, keep_order=True)
except ValueError:
    # Create the database from GFF if it does not exist
    db = gffutils.create_db(
        gff_file,
        dbfn=db_file,
        force=True,
        keep_order=True,
        merge_strategy='merge',
        sort_attribute_values=True
    )

# --------------------------------------------------------------------------------
# Load and filter data
# --------------------------------------------------------------------------------
def load_and_filter_data(file_path, strand):
    df = pd.read_csv(file_path, sep='\t')
    
    if strand == 'forward':
        df_filtered = df[
            (df['Log2_FC'] >= 1.8) & 
            (df['WT_Ratio'] >= 4) &
            (df['WT_Forward_Upstream_Median'] >= 0.03) &
            (df['dRho_Forward_Upstream_Median'] >= 0.03)
        ].copy()

    elif strand == 'reverse':
        df_filtered = df[
            (df['Log2_FC'] >= 1.8) & 
            (df['WT_Ratio'] >= 4) &
            (df['WT_Reverse_Downstream_Median'] >= 0.03) &
            (df['dRho_Reverse_Downstream_Median'] >= 0.03)
        ].copy()

    else:
        raise ValueError("Strand must be 'forward' or 'reverse'")
    
    return df_filtered

# --------------------------------------------------------------------------------
# Identify Overlaps
# --------------------------------------------------------------------------------
def identify_overlaps(df, min_distance):
    overlaps = []
    current_group = 0
    last_coord = df.iloc[0]['Coord'] - min_distance

    for index, row in df.iterrows():
        if (row['Coord'] - last_coord) >= min_distance:
            current_group += 1
        overlaps.append(current_group)
        last_coord = row['Coord']
    
    df['Overlap_Group'] = overlaps
    return df

# --------------------------------------------------------------------------------
# Merge Pseudo Clusters
# --------------------------------------------------------------------------------
def merge_pseudo_clusters(df):
    df.sort_values('Coord', inplace=True)

    df_clusters = df.groupby('Overlap_Group').agg({
        'Coord': ['min', 'max'],
        'WT_Ratio': ['max', 'mean'],
        'Log2_FC': ['max', 'mean']
    }).reset_index()

    df_clusters.columns = [
        'Overlap_Group', 'Start', 'End',
        'Max_WT_Ratio', 'Average_WT_Ratio',
        'Max_Log2_FC', 'Average_Log2_FC'
    ]

    df_clusters['Length'] = df_clusters['End'] - df_clusters['Start']
    df_clusters['Distance_To_Next'] = df_clusters['Start'].shift(-1) - df_clusters['End']
    df_clusters['Merge_Flag'] = (
        (df_clusters['Distance_To_Next'] < 60) &
        (df_clusters['Distance_To_Next'].notna())
    )

    new_group = 0
    groups = []
    for i in range(len(df_clusters)):
        if i == 0 or not df_clusters.iloc[i-1, df_clusters.columns.get_loc('Merge_Flag')]:
            new_group += 1
        groups.append(new_group)

    df_clusters['New_Overlap_Group'] = groups

    df_merged = df_clusters.groupby('New_Overlap_Group').agg({
        'Start': 'min',
        'End': 'max',
        'Max_WT_Ratio': 'max',
        'Average_WT_Ratio': 'mean',
        'Max_Log2_FC': 'max',
        'Average_Log2_FC': 'mean',
        'Length': 'max'
    }).reset_index()

    group_counts = df_clusters.groupby('New_Overlap_Group')['Overlap_Group'].nunique()
    df_merged['Split/Merged'] = df_merged['New_Overlap_Group'].map(group_counts)
    df_merged['Split/Merged'] = df_merged['Split/Merged'].apply(lambda x: 'Yes' if x > 1 else 'No')

    return df_merged

# --------------------------------------------------------------------------------
# Calculate Cluster Statistics
# --------------------------------------------------------------------------------
def calculate_cluster_statistics(df_clusters, df, chrom):
    cluster_stats = []

    for _, cluster in df_clusters.iterrows():
        start = cluster['Start']
        end   = cluster['End']
        split_merged = cluster['Split/Merged']

        # Extract the data corresponding to the current cluster range
        cluster_data = df[(df['Coord'] >= start) & (df['Coord'] <= end)]
        
        # Calculate max and average statistics for WT Ratio and Log2_FC
        max_toxic_ratio = cluster_data['WT_Ratio'].max()
        avg_toxic_ratio = cluster_data['WT_Ratio'].mean()
        max_log2fc      = cluster_data['Log2_FC'].max()
        avg_log2fc      = cluster_data['Log2_FC'].mean()

        # Coordinates for Max_WT_Ratio
        max_wt_ratio_coords = cluster_data[cluster_data['WT_Ratio'] == max_toxic_ratio]['Coord'].values
        if len(max_wt_ratio_coords) > 1:
           
            if max_wt_ratio_coords[-1] - max_wt_ratio_coords[0] == len(max_wt_ratio_coords) - 1:
                max_wt_ratio_coord = int((max_wt_ratio_coords[0] + max_wt_ratio_coords[-1]) / 2)
            else:
                max_wt_ratio_coord = max_wt_ratio_coords[0]
        else:
            max_wt_ratio_coord = max_wt_ratio_coords[0]

        # Coordinates for Max_log2_FC
        max_log2fc_coords = cluster_data[cluster_data['Log2_FC'] == max_log2fc]['Coord'].values
        if len(max_log2fc_coords) > 1:
          
            if max_log2fc_coords[-1] - max_log2fc_coords[0] == len(max_log2fc_coords) - 1:
                max_log2fc_coord = int((max_log2fc_coords[0] + max_log2fc_coords[-1]) / 2)
            else:
                max_log2fc_coord = max_log2fc_coords[0]
        else:
            max_log2fc_coord = max_log2fc_coords[0]

        cluster_stats.append({
            'Start': start,
            'Center': int((start + end) / 2),
            'End': end,
            'Split/Merged': split_merged,
            'Max_WT_Ratio': max_toxic_ratio,
            'Average_WT_Ratio': avg_toxic_ratio,
            'Max_WT_Ratio_Coord': max_wt_ratio_coord,
            'Max_log2_FC': max_log2fc,
            'Average_log2_FC': avg_log2fc,
            'Max_log2_FC_Coord': max_log2fc_coord,
            'chrom': chrom
        })

    return pd.DataFrame(cluster_stats)

# --------------------------------------------------------------------------------
# Gene/TSS Info
# --------------------------------------------------------------------------------
def get_gene_info(db, clusters, tss_df, strand):
    clusters['Gene'] = ''
    clusters['Sense_or_Anti'] = ''
    clusters['Intergenic_or_AKA_Code'] = ''
    clusters['Closest_TSS'] = None
    clusters['Closest_ATG'] = None

    # Filter out pseudogenes
    all_genes = list(db.all_features(featuretype='gene'))
    non_pseudogenes = [
        g for g in all_genes
        if 'pseudogene' not in g.attributes.get('type', '')
    ]

    for index, row in clusters.iterrows():
        chrom = row['chrom']
        start = row['Start']
        end   = row['End']

        # Overlapping non-pseudogene genes
        overlapping_genes = [
            g for g in non_pseudogenes
            if g.seqid == chrom and not (g.end < start or g.start > end)
        ]

        if overlapping_genes:
            gene = overlapping_genes[0]
            # Sense or anti
            if gene.strand == row['Strand']:
                clusters.at[index, 'Sense_or_Anti'] = 'Sense'
            else:
                clusters.at[index, 'Sense_or_Anti'] = 'Anti'

            clusters.at[index, 'Gene'] = (
                gene['Name'][0] if 'Name' in gene.attributes else gene.id
            )
            clusters.at[index, 'Intergenic_or_AKA_Code'] = ''
        else:
            # Check pseudogene overlap
            pseudo_overlap = [
                g for g in db.all_features(featuretype='pseudogene')
                if g.seqid == chrom and not (g.end < start or g.start > end)
            ]
            if pseudo_overlap:
                clusters.at[index, 'Gene'] = 'Intergenic'
                clusters.at[index, 'Sense_or_Anti'] = ''
                clusters.at[index, 'Intergenic_or_AKA_Code'] = 'Intergenic'
            else:
               
                upstream_genes = [
                    g for g in non_pseudogenes
                    if g.seqid == chrom and g.end < start
                ]
                upstream_genes = sorted(
                    upstream_genes, key=lambda x: x.end, reverse=True
                )[:1]

                downstream_genes = [
                    g for g in non_pseudogenes
                    if g.seqid == chrom and g.start > end
                ]
                downstream_genes = sorted(
                    downstream_genes, key=lambda x: x.start
                )[:1]

                if upstream_genes and 'Name' in upstream_genes[0].attributes:
                    closest_gene_upstream = upstream_genes[0]['Name'][0]
                else:
                    closest_gene_upstream = 'NA'
                if downstream_genes and 'Name' in downstream_genes[0].attributes:
                    closest_gene_downstream = downstream_genes[0]['Name'][0]
                else:
                    closest_gene_downstream = 'NA'

                intergenic_code = f"{closest_gene_upstream},{closest_gene_downstream}"
                clusters.at[index, 'Gene'] = 'Intergenic'
                clusters.at[index, 'Sense_or_Anti'] = ''
                clusters.at[index, 'Intergenic_or_AKA_Code'] = intergenic_code

        # Closest TSS
        if strand == 'forward':
            relevant_tss = tss_df[tss_df['TSS_direction'] == '+']
        elif strand == 'reverse':
            relevant_tss = tss_df[tss_df['TSS_direction'] == '-']
        else:
            raise ValueError("Strand must be 'forward' or 'reverse'")

        if not relevant_tss.empty:
            closest_tss_idx = (relevant_tss['TSS_site'] - start).abs().idxmin()
            clusters.at[index, 'Closest_TSS'] = relevant_tss.loc[closest_tss_idx, 'TSS_site']

        # Closest ATG
        cds_positions = [
            c.start if c.strand == row['Strand'] else c.end
            for c in db.features_of_type('CDS')
            if c.seqid == chrom and c.strand == row['Strand']
        ]
        if cds_positions:
            closest_atg = min(cds_positions, key=lambda pos: abs(pos - start))
            clusters.at[index, 'Closest_ATG'] = closest_atg

    return clusters

# --------------------------------------------------------------------------------
# CT and G content calculations
# --------------------------------------------------------------------------------
def calculate_ct_content(sequence):
    if len(sequence) == 0:
        return 0
    ct_count = sequence.count('C') + sequence.count('T')
    return ct_count / len(sequence)

def sliding_window_ct_content(sequence, base_window_size=30):
    results = []
    for i in range(len(sequence) - base_window_size + 1):
        window = sequence[i:i + base_window_size]
        ct_content = (window.count('C') + window.count('T')) / len(window)
        results.append(ct_content)
    return results

def sliding_window_g_content(sequence, base_window_size=30):
    results = []
    for i in range(len(sequence) - base_window_size + 1):
        window = sequence[i:i + base_window_size]
        g_content = window.count('G') / len(window)
        results.append(g_content)
    return results

# --------------------------------------------------------------------------------
# Find highest CT or lowest G region in Upstream / Downstream
# --------------------------------------------------------------------------------
def find_highest_ct_content_region(upstream_seq, downstream_seq, start_coord):
    """
    Use sliding windows of length base_window_size=30 to find the coordinate
    where CT content is highest in upstream and downstream sequences.
    """
    upstream_ct_windows = sliding_window_ct_content(upstream_seq, base_window_size=30)
    downstream_ct_windows = sliding_window_ct_content(downstream_seq, base_window_size=30)
    
    # upstream
    if upstream_ct_windows:
        max_up_idx = upstream_ct_windows.index(max(upstream_ct_windows))
        upstream_center = start_coord - len(upstream_seq) + max_up_idx + 15
    else:
        upstream_center = None
    
    # downstream
    if downstream_ct_windows:
        max_down_idx = downstream_ct_windows.index(max(downstream_ct_windows))
        downstream_center = start_coord + max_down_idx + 15
    else:
        downstream_center = None
    
    return upstream_center, downstream_center

def find_lowest_g_content_region(upstream_seq, downstream_seq, start_coord):
    """
    Use sliding windows of length base_window_size=30 to find the coordinate
    where G content is lowest in upstream and downstream sequences.
    """
    upstream_g_windows   = sliding_window_g_content(upstream_seq, base_window_size=30)
    downstream_g_windows = sliding_window_g_content(downstream_seq, base_window_size=30)
    
    # upstream
    if upstream_g_windows:
        min_up_idx = upstream_g_windows.index(min(upstream_g_windows))
        upstream_center = start_coord - len(upstream_seq) + min_up_idx + 15
    else:
        upstream_center = None
    
    # downstream
    if downstream_g_windows:
        min_down_idx = downstream_g_windows.index(min(downstream_g_windows))
        downstream_center = start_coord + min_down_idx + 15
    else:
        downstream_center = None
    
    return upstream_center, downstream_center

# --------------------------------------------------------------------------------
# 1) Combine Upstream + Downstream for a Single 200-nt Sequence
# --------------------------------------------------------------------------------
def calculate_combined_windows(upstream_seq, downstream_seq, base_window_size=30):

    combined_seq = upstream_seq + downstream_seq  
    ct_values = sliding_window_ct_content(combined_seq, base_window_size=base_window_size)
    g_values  = sliding_window_g_content(combined_seq, base_window_size=base_window_size)
    return ct_values, g_values

# --------------------------------------------------------------------------------
# 2) Expand a list-type column into multiple columns (1..171, etc.)
# --------------------------------------------------------------------------------
def expand_window_columns(df, window_column, prefix):
    expanded_cols = pd.DataFrame(df[window_column].tolist(), index=df.index)
    expanded_cols.columns = [f'{prefix}{i+1}' for i in range(expanded_cols.shape[1])]
    return expanded_cols

# --------------------------------------------------------------------------------
# 3) Main function to calculate additional features for each cluster
# --------------------------------------------------------------------------------
def calculate_additional_features(df):
   
    # Basic stats
    df['Cluster_Length'] = df['End'] - df['Start']
    df['CT_Upstream_Content']   = df['Upstream_From_Start_Seq'].apply(calculate_ct_content)
    df['CT_Downstream_Content'] = df['Downstream_From_Start_Seq'].apply(calculate_ct_content)
    
    # Find highest CT and lowest G in separate Up/Down
    df[['Highest_CT_Content_Upstream_Coord','Highest_CT_Content_Downstream_Coord']] = df.apply(
        lambda row: find_highest_ct_content_region(
            row['Upstream_From_Start_Seq'],
            row['Downstream_From_Start_Seq'],
            row['Start']
        ),
        axis=1, result_type="expand"
    )
    df[['Lowest_G_Content_Upstream_Coord','Lowest_G_Content_Downstream_Coord']] = df.apply(
        lambda row: find_lowest_g_content_region(
            row['Upstream_From_Start_Seq'],
            row['Downstream_From_Start_Seq'],
            row['Start']
        ),
        axis=1, result_type="expand"
    )
    
    combined_results = df.apply(
        lambda row: calculate_combined_windows(
            row['Upstream_Seq'], 
            row['Downstream_Seq'], 
            base_window_size=30
        ),
        axis=1
    )
    df['CT_Combined_Window'] = combined_results.apply(lambda x: x[0])
    df['G_Combined_Window']  = combined_results.apply(lambda x: x[1])
    
    # Expand into contented windows
    df_ct_expanded = expand_window_columns(df, 'CT_Combined_Window', 'CT_Combined_Window')
    df_g_expanded  = expand_window_columns(df, 'G_Combined_Window',  'G_Combined_Window')
    
    df = pd.concat([df, df_ct_expanded, df_g_expanded], axis=1)
    
    df.drop(columns=['CT_Combined_Window','G_Combined_Window'], inplace=True)

    return df

# --------------------------------------------------------------------------------
# 4) Process Each Strand
# --------------------------------------------------------------------------------
def process_strand(file_path, strand, db, tss_df):
    df_filtered = load_and_filter_data(file_path, strand)
    
    # Identify overlaps, cluster, stats
    df_overlaps = identify_overlaps(df_filtered, min_distance=20)
    df_clusters = merge_pseudo_clusters(df_overlaps)
    print(f"Initial number of clusters ({strand}):", len(df_clusters))
    print(df_clusters[['Start', 'End', 'Split/Merged']].head())
    
    df_final = calculate_cluster_statistics(df_clusters, df_filtered, chrom='NC_000913.2')
    print(df_final.head())
    
    df_final['chrom'] = 'NC_000913.2'
    
    # Load reference genome
    reference_genome = SeqIO.read("NC_000913.2.fasta", "fasta").seq
    
    # Create columns for sequences around the cluster center (±100 nt),
    # and from cluster start (±100 nt).
    if strand == 'forward':
        df_final['Upstream_Seq'] = df_final['Center'].apply(lambda x: str(reference_genome[x-100:x]))
        df_final['Downstream_Seq'] = df_final['Center'].apply(lambda x: str(reference_genome[x:x+100]))
        df_final['Upstream_From_Start_Seq'] = df_final['Start'].apply(lambda x: str(reference_genome[x-100:x]))
        df_final['Downstream_From_Start_Seq'] = df_final['Start'].apply(lambda x: str(reference_genome[x:x+100]))
        df_final['Strand'] = '+'
    else:  # strand == 'reverse'
        df_final['Upstream_Seq'] = df_final['Center'].apply(
            lambda x: str(reference_genome[x:x+100].reverse_complement())
        )
        df_final['Downstream_Seq'] = df_final['Center'].apply(
            lambda x: str(reference_genome[x-100:x].reverse_complement())
        )
        df_final['Upstream_From_Start_Seq'] = df_final['Start'].apply(
            lambda x: str(reference_genome[x:x+100].reverse_complement())
        )
        df_final['Downstream_From_Start_Seq'] = df_final['Start'].apply(
            lambda x: str(reference_genome[x-100:x].reverse_complement())
        )
        df_final['Strand'] = '-'
    
    df_final = get_gene_info(db, df_final, tss_df, strand)
    
    df_final = calculate_additional_features(df_final)

    return df_final

# --------------------------------------------------------------------------------
# Main Execution
# --------------------------------------------------------------------------------

# Process forward and reverse
forward_df = process_strand(forward_file, 'forward', db, tss_df)
reverse_df = process_strand(reverse_file, 'reverse', db, tss_df)

# Combine results
combined_df = pd.concat([forward_df, reverse_df], ignore_index=True)
combined_df = combined_df.sort_values(by='Start').reset_index(drop=True)

output_columns = [
    'Start', 'Center', 'End', 'Strand', 'Split/Merged',
    'Max_WT_Ratio', 'Max_WT_Ratio_Coord', 'Average_WT_Ratio',
    'Max_log2_FC', 'Max_log2_FC_Coord', 'Average_log2_FC',
    'Upstream_Seq', 'Downstream_Seq', 
    'Upstream_From_Start_Seq', 'Downstream_From_Start_Seq',
    'Gene', 'Sense_or_Anti', 'Intergenic_or_AKA_Code',
    'Cluster_Length', 'CT_Upstream_Content', 'CT_Downstream_Content',
    'Closest_TSS', 'Closest_ATG',
    'Highest_CT_Content_Upstream_Coord','Highest_CT_Content_Downstream_Coord',
    'Lowest_G_Content_Upstream_Coord','Lowest_G_Content_Downstream_Coord'
]

# expanded columns for the CT/G combined windows
combined_window_cols = [
    col for col in combined_df.columns
    if col.startswith('CT_Combined_Window') or col.startswith('G_Combined_Window')
]
final_columns = output_columns + combined_window_cols

# Save to CSV
output_file = 'combined_strands_211_212_213_TOX1328_vs_234_235_236_1325_BCM_m14_M30_clusters_identified.csv'
combined_df.to_csv(output_file, columns=final_columns, index=False)
print(f"Combined results saved to {output_file}")