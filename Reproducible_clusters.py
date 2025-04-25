import pandas as pd
import numpy as np
import os
import re
import gffutils
from sklearn.cluster import DBSCAN
from Bio import SeqIO


###########################
# 1) LOAD GFF/DB
###########################
tss_file_path = 'TSS.csv'
tss_df = pd.read_csv(tss_file_path)
gff_file = 'NC_000913.2.gff3'
db_file = 'NC_000913.2.db'

try:
    db = gffutils.FeatureDB(db_file, keep_order=True)
    print(f"Loaded existing database: {db_file}")
except ValueError:
    print(f"Creating database {db_file} from {gff_file}...")
    db = gffutils.create_db(
        gff_file,
        dbfn=db_file,
        force=True,
        keep_order=True,
        merge_strategy='merge',
        sort_attribute_values=True
    )
    print("Database created.")

###########################
# 2) HELPER FUNCTIONS
###########################
def calculate_ct_content(sequence):
    if not sequence:
        return 0
    return (sequence.count('C') + sequence.count('T')) / len(sequence)
    
#calculate CT content for all 30-nt windows in upstream and downstream sequences
def sliding_window_ct_content(sequence, window_size=30):
    results = []
    for i in range(len(sequence) - window_size + 1):
        window = sequence[i:i + window_size]
        ct_content = (window.count('C') + window.count('T')) / len(window)
        results.append(ct_content)
    return results

#calculate G content for all 30-nt windows in upstream and downstream sequences
def sliding_window_g_content(sequence, window_size=30):
    results = []
    for i in range(len(sequence) - window_size + 1):
        window = sequence[i:i + window_size]
        g_content = window.count('G') / len(window)
        results.append(g_content)
    return results


#identifies the genomic coordinates of regions with the highest CT content
def find_highest_ct_content_region(upstream_seq, downstream_seq, start_coord):
    upstream_ct = sliding_window_ct_content(upstream_seq, 30)
    downstream_ct = sliding_window_ct_content(downstream_seq, 30)

    if upstream_ct:
        max_upstream_ct_index = upstream_ct.index(max(upstream_ct))
        upstream_region_center = start_coord - len(upstream_seq) + max_upstream_ct_index + 15
    else:
        upstream_region_center = np.nan

    if downstream_ct:
        max_downstream_ct_index = downstream_ct.index(max(downstream_ct))
        downstream_region_center = start_coord + max_downstream_ct_index + 15
    else:
        downstream_region_center = np.nan

    return upstream_region_center, downstream_region_center

#identifies the genomic coordinates of regions with the lowest G content
def find_lowest_g_content_region(upstream_seq, downstream_seq, start_coord):
    upstream_g = sliding_window_g_content(upstream_seq, 30)
    downstream_g = sliding_window_g_content(downstream_seq, 30)

    if upstream_g:
        min_upstream_g_index = upstream_g.index(min(upstream_g))
        upstream_region_center = start_coord - len(upstream_seq) + min_upstream_g_index + 15
    else:
        upstream_region_center = np.nan

    if downstream_g:
        min_downstream_g_index = downstream_g.index(min(downstream_g))
        downstream_region_center = start_coord + min_downstream_g_index + 15
    else:
        downstream_region_center = np.nan

    return upstream_region_center, downstream_region_center

#extracts genomic sequences around a cluster position, accounting for strand orientation
def build_sequences_for_row(row, genome, window_size=100):
    s = row['Strand']
    center = int(row['Center'])
    start = int(row['Start'])
    
 
    up_center_start = max(center - window_size, 0)
    up_center_end   = center
    down_center_start = center
    down_center_end   = center + window_size

    up_start_start = max(start - window_size, 0)
    up_start_end   = start
    down_start_start = start
    down_start_end   = start + window_size

    full_start = max(center - 100, 0)
    full_end   = center + 100  

    if s == '+':
        upstream_seq = genome[up_center_start:up_center_end]
        downstream_seq = genome[down_center_start:down_center_end]
        up_from_start_seq = genome[up_start_start:up_start_end]
        down_from_start_seq = genome[down_start_start:down_start_end]
        full_seq = genome[full_start:full_end]
    else:
        # For - strand
        upstream_seq = genome[center:center+window_size].reverse_complement()
        downstream_seq = genome[center-window_size:center].reverse_complement()
        up_from_start_seq = genome[start:start+window_size].reverse_complement()
        down_from_start_seq = genome[start-window_size:start].reverse_complement()
        
        full_region = genome[full_start:full_end]
        full_seq = full_region.reverse_complement()

    return {
        'Upstream_Seq': str(upstream_seq),
        'Downstream_Seq': str(downstream_seq),
        'Upstream_From_Start_Seq': str(up_from_start_seq),
        'Downstream_From_Start_Seq': str(down_from_start_seq),
        'Full_Seq': str(full_seq)  
    }

def expand_window_columns(df, window_column, prefix):
 
    expanded_cols = pd.DataFrame(df[window_column].tolist(), index=df.index)
    expanded_cols.columns = [f'{prefix}{i+1}' for i in range(expanded_cols.shape[1])]
    return expanded_cols

def get_gene_info(db, clusters, tss_df, strand_mode):
    clusters['Gene'] = ''
    clusters['Sense_or_Anti'] = ''
    clusters['Intergenic_or_AKA_Code'] = ''
    clusters['Closest_TSS'] = None
    clusters['Closest_ATG'] = None

    all_genes = list(db.all_features(featuretype='gene'))
    non_pseudogenes = [g for g in all_genes if 'pseudogene' not in g.attributes.get('type', '')]

    for index, row in clusters.iterrows():
        chrom = row['chrom']
        start = row['Start']
        end   = row['End']
        overlapping_genes = [
            gene for gene in non_pseudogenes
            if gene.seqid == chrom and not (gene.end < start or gene.start > end)
        ]

        if overlapping_genes:
            gene = overlapping_genes[0]
            if gene.strand == row['Strand']:
                clusters.at[index, 'Sense_or_Anti'] = 'Sense'
            else:
                clusters.at[index, 'Sense_or_Anti'] = 'Anti'

            clusters.at[index, 'Gene'] = (
                gene['Name'][0] if 'Name' in gene.attributes else gene.id
            )
            clusters.at[index, 'Intergenic_or_AKA_Code'] = ''
        else:
            # Check pseudogene
            pseudo_overlaps = [
                g for g in db.all_features(featuretype='pseudogene')
                if g.seqid == chrom and not (g.end < start or g.start > end)
            ]
            if pseudo_overlaps:
                clusters.at[index, 'Gene'] = 'Intergenic'
                clusters.at[index, 'Sense_or_Anti'] = ''
                clusters.at[index, 'Intergenic_or_AKA_Code'] = 'Intergenic'
            else:
                # truly intergenic
                upstream_genes = [
                    g for g in non_pseudogenes
                    if g.seqid == chrom and g.end < start
                ]
                upstream_genes = sorted(upstream_genes, key=lambda g: g.end, reverse=True)[:1]
                downstream_genes = [
                    g for g in non_pseudogenes
                    if g.seqid == chrom and g.start > end
                ]
                downstream_genes = sorted(downstream_genes, key=lambda g: g.start)[:1]

                if upstream_genes and 'Name' in upstream_genes[0].attributes:
                    up_name = upstream_genes[0]['Name'][0]
                else:
                    up_name = 'NA'
                if downstream_genes and 'Name' in downstream_genes[0].attributes:
                    down_name = downstream_genes[0]['Name'][0]
                else:
                    down_name = 'NA'

                intergenic_code = f"{up_name},{down_name}"
                clusters.at[index, 'Gene'] = 'Intergenic'
                clusters.at[index, 'Sense_or_Anti'] = ''
                clusters.at[index, 'Intergenic_or_AKA_Code'] = intergenic_code

        # TSS
        if strand_mode == 'forward':
            relevant_tss = tss_df[tss_df['TSS_direction'] == '+']
        elif strand_mode == 'reverse':
            relevant_tss = tss_df[tss_df['TSS_direction'] == '-']
        else:
            relevant_tss = pd.DataFrame()
        if not relevant_tss.empty:
            closest_tss_idx = (relevant_tss['TSS_site'] - start).abs().idxmin()
            closest_tss = relevant_tss.loc[closest_tss_idx, 'TSS_site']
            clusters.at[index, 'Closest_TSS'] = closest_tss

        # Closest ATG
        cds_positions = [
            (cds.start if cds.strand == '+' else cds.end)
            for cds in db.features_of_type('CDS')
            if cds.seqid == chrom and cds.strand == row['Strand']
        ]
        if cds_positions:
            closest_atg = min(cds_positions, key=lambda pos: abs(pos - start))
            clusters.at[index, 'Closest_ATG'] = closest_atg

    return clusters

###########################
# 3) READ & COMBINE CSVs
###########################

# File paths
file_paths = [
    'combined_strands_1325_WT_199_200_201_vs_202_203_209_1325BCM_clusters_identified_100.csv',
    'combined_strands_1325_WT_199_200_201_vs_230_231_232_DnG_Ara_m14_M30_clusters_identified_100.csv',
    'combined_strands_1325_WT_199_200_201_vs_243_235_236_1325_BCM_m14_M30_clusters_identified_100.csv',
    'combined_strands_1325_WT_199_200_201_vs_1324_dRho_204_205_206_clusters_identified_100.csv',
    'combined_strands_1325_WT_199_200_201_vs_dRho_1261_Ara_92_101_102_m14_M30_clusters_identified_100.csv',
    'combined_strands_1325_WT_199_200_201_vs_LLJ_LLKK_179_m19_M30_clusters_identified_100.csv',
    'combined_strands_1325_WT_199_200_201_vs_WT_1246_LL_Eb_2BQ_2BG_m14_M30_clusters_identified_100.csv',
    'combined_strands_Tox1328_vs_BCM_clusters_identified_100.csv',
    'combined_strands_WT_1246_LL_Eb_Q_G_m14_M30_vs_dRho_1261_Ara_92_101_102_m14_M30_clusters_identified_100.csv'
]

df_list = []
dataset_names = []  # Store dataset names here

for path in file_paths:
    match = re.search(r'combined_strands_(.+?)_clusters_identified_100', path)
    if match:
        dname = match.group(1)
    else:
        dname = os.path.splitext(os.path.basename(path))[0]
    
    dataset_names.append(dname) 

    tmp = pd.read_csv(path)
    tmp['Dataset'] = dname
    df_list.append(tmp)

# Combine all datasets into one DataFrame
combined_data = pd.concat(df_list, ignore_index=True)



###########################
# 4) DBSCAN CLUSTERING
###########################

results = []
for strand in ['+', '-']:
    strand_data = combined_data[combined_data['Strand'] == strand].copy()
    if strand_data.empty:
        continue

    coords = strand_data[['Center']].values
    model = DBSCAN(eps=100, min_samples=1, metric='manhattan').fit(coords)
    strand_data['cluster_id'] = model.labels_

    cluster_summaries = []
    for cid, group in strand_data.groupby('cluster_id'):
        row = {
            'cluster_id': cid,
            'Strand': strand,
            'n_calls': len(group),
            'furthest_start': group['Start'].min(),
            'furthest_end': group['End'].max(),
            'reproducibility': (
                'Reproducible' if len(group['Dataset'].unique()) > 1 else 'Unique'
            ),
            'datasets_involved': ','.join(group['Dataset'].unique())
        }
        for dname in dataset_names:
            sub = group[group['Dataset'] == dname]
            row[f'{dname}_positions'] = (
                ';'.join(map(str, sorted(sub['Center']))) if not sub.empty else ''
            )
        cluster_summaries.append(row)

    strand_summary_df = pd.DataFrame(cluster_summaries)
    results.append(strand_summary_df)

final_data = pd.concat(results, ignore_index=True)

final_data.rename(columns={
    'furthest_start': 'Start',
    'furthest_end': 'End'
}, inplace=True)
final_data['Center'] = (final_data['Start'] + final_data['End']) // 2
final_data['chrom'] = 'NC_000913.2'

###########################
# 5) ATTACH SEQUENCES
###########################
reference_genome = SeqIO.read("NC_000913.2.fasta", "fasta").seq

# Build sequences
seq_data = final_data.apply(
    lambda row: build_sequences_for_row(row, reference_genome, 100),
    axis=1, result_type='expand'
)
final_data = pd.concat([final_data, seq_data], axis=1)

###########################
# 6) GENE / TSS / ATG ANNOTATION
###########################
plus_df  = final_data[final_data['Strand'] == '+'].copy()
minus_df = final_data[final_data['Strand'] == '-'].copy()

plus_annotated  = get_gene_info(db, plus_df,  tss_df, strand_mode='forward')
minus_annotated = get_gene_info(db, minus_df, tss_df, strand_mode='reverse')

annotated_df = pd.concat([plus_annotated, minus_annotated], ignore_index=True)

###########################
# 7) ADDITIONAL FEATURES
###########################
def calculate_additional_features(df):
    # Keep existing cluster length
    df['Cluster_Length'] = df['End'] - df['Start']

    # Basic CT content on upstream/downstream
    df['CT_Upstream_Content'] = df['Upstream_From_Start_Seq'].apply(calculate_ct_content)
    df['CT_Downstream_Content'] = df['Downstream_From_Start_Seq'].apply(calculate_ct_content)

    # Highest/lowest region calls remain the same
    df[['Highest_CT_Content_Upstream_Coord', 'Highest_CT_Content_Downstream_Coord']] = df.apply(
        lambda row: find_highest_ct_content_region(
            row['Upstream_From_Start_Seq'], row['Downstream_From_Start_Seq'], row['Start']
        ),
        axis=1, result_type='expand'
    )
    df[['Lowest_G_Content_Upstream_Coord', 'Lowest_G_Content_Downstream_Coord']] = df.apply(
        lambda row: find_lowest_g_content_region(
            row['Upstream_From_Start_Seq'], row['Downstream_From_Start_Seq'], row['Start']
        ),
        axis=1, result_type='expand'
    )

    df['CT_Full_Window'] = df['Full_Seq'].apply(sliding_window_ct_content)
    df['G_Full_Window']  = df['Full_Seq'].apply(sliding_window_g_content)

    return df

annotated_df = calculate_additional_features(annotated_df)

# expand the "Full" windows into columns
full_ct_expanded = expand_window_columns(annotated_df, 'CT_Full_Window', 'CT_Full_Window')
full_g_expanded  = expand_window_columns(annotated_df, 'G_Full_Window', 'G_Full_Window')

annotated_df = pd.concat([annotated_df, full_ct_expanded, full_g_expanded], axis=1)

annotated_df.drop(columns=['CT_Full_Window', 'G_Full_Window'], inplace=True)

###########################
# 8) FINAL OUTPUT
###########################
dataset_position_cols = [f'{d}_positions' for d in dataset_names]

output_cols = [
    'Strand','Start','End','Center',
    'reproducibility','datasets_involved'
] + dataset_position_cols + [
    # Keep the gene info
    'Gene','Sense_or_Anti','Intergenic_or_AKA_Code',
    'Closest_TSS','Closest_ATG',
    # Keep sequences
    'Upstream_Seq','Downstream_Seq',
    'Upstream_From_Start_Seq','Downstream_From_Start_Seq',
    'Full_Seq', 
    # Keep cluster length & content
    'Cluster_Length','CT_Upstream_Content','CT_Downstream_Content',
    'Highest_CT_Content_Upstream_Coord','Highest_CT_Content_Downstream_Coord',
    'Lowest_G_Content_Upstream_Coord','Lowest_G_Content_Downstream_Coord'
]

expanded_window_cols = [
    c for c in annotated_df.columns
    if c.startswith('CT_Full_Window') or c.startswith('G_Full_Window')
]
output_cols += expanded_window_cols

final_cols = [c for c in output_cols if c in annotated_df.columns]
annotated_df.sort_values(by='Start', inplace=True, ignore_index=True)

annotated_df[final_cols].to_csv("reproducible_unique_clusters.csv", index=False)
print("Done. Output saved to reproducible_unique_clusters.csv")