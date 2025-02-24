import pandas as pd
import numpy as np

#this script is used after the coverage.py script. Once the 
# median coverages have been calculated for the upstream and downstream of each nucleotide
# and the ratios have been calculated.. this script is used for the 
# log2FC

np.seterr(divide='ignore', invalid='ignore')




def calculate_log2_fc_with_additional_columns(wt_df, dRho_df, ratio_column, additional_columns):
    # Ensure the DataFrame indexes match for accurate row-wise operation
    wt_df.set_index('Coord', inplace=True)
    dRho_df.set_index('Coord', inplace=True)

    # Calculate log2 fold change (FC)
    fc_df = pd.DataFrame(index=wt_df.index)
    fc_df['WT_Ratio'] = wt_df[ratio_column]
    fc_df['dRho_Ratio'] = dRho_df[ratio_column]
    fc_df['Log2_FC'] = -np.log2(fc_df['dRho_Ratio'] / fc_df['WT_Ratio'])

    # Handle cases where division by zero or log of zero might occur
    fc_df.replace([np.inf, -np.inf, np.nan], 0, inplace=True)
    
    # Add additional columns from the original dataframes
    for col in additional_columns:
        fc_df[f'WT_{col}'] = wt_df[col]
        fc_df[f'dRho_{col}'] = dRho_df[col]

    
    return fc_df

# Specify the additional columns you're interested in
additional_columns_forward = ['Forward_Upstream_Median', 'Forward_Downstream_Median']
additional_columns_reverse = ['Reverse_Upstream_Median', 'Reverse_Downstream_Median']

# Load the ratio files for toxic peptide and BCM strains for forward and reverse strands
wt_forward_df = pd.read_csv('1325_WT_199_200_201_all_files/1325_WT_199_200_201_fwd_coverage.txt', sep='\t')
dRho_forward_df = pd.read_csv('243_235_236_1325_BCM_m14_M30_all_files/243_235_236_1325_BCM_m14_M30_fwd_coverage.txt', sep='\t')
wt_reverse_df = pd.read_csv('1325_WT_199_200_201_all_files/1325_WT_199_200_201_rev_coverage.txt', sep='\t')
dRho_reverse_df = pd.read_csv('243_235_236_1325_BCM_m14_M30_all_files/243_235_236_1325_BCM_m14_M30_rev_coverage.txt', sep='\t')

# Calculate log2 fold change for forward and reverse strands, specifying the correct ratio column and the additional columns
forward_fc_df = calculate_log2_fc_with_additional_columns(wt_forward_df, dRho_forward_df, 'Forward_Ratio', additional_columns_forward)
reverse_fc_df = calculate_log2_fc_with_additional_columns(wt_reverse_df, dRho_reverse_df, 'Reverse_Ratio', additional_columns_reverse)

# Save the differential tables to separate files for forward and reverse strands
forward_fc_df.to_csv('1325_WT_199_200_201_vs_243_235_236_1325_BCM_m14_M30_forward_strand_differential_table.txt', sep='\t', float_format='%.3f')
reverse_fc_df.to_csv('1325_WT_199_200_201_vs_243_235_236_1325_BCM_m14_M30_reverse_strand_differential_table.txt', sep='\t', float_format='%.3f')

print("The differential log2 fold change tables have been saved.")
