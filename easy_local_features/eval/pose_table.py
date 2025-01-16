import os
import glob
import json
import pandas as pd
import sys
import argparse
from pathlib import Path

FOLDER_RESULTS = "./output/pose_eval/scannet"
def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",  "-o", help="Output folder", default=FOLDER_RESULTS)
    parser.add_argument("--accuracy",  "-acc", action='store_true', help="Print accuracy")
    parser.add_argument("--filter",  "-f", nargs='+', help="Filter by name")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse()
    
    dataset_name = 'scannet'
    all_summary_files = list(Path(args.output).glob("*/*_summary.json"))
    all_summary_files = [str(f) for f in all_summary_files]

    if len(all_summary_files) == 0:
        print(f"No summary files found at {args.output}")
        exit(1)

    filtering = []
    if args.filter:
        filtering = args.filter
        # all_summary_files = [f for f in all_summary_files if filtering in f]
        # include if in any of the filtering
        all_summary_files = [f for f in all_summary_files if any([fil in f for fil in filtering])]

    dfs = []
    names = []
    estimators = []
    metric_key = 'aucs_by_thresh'
    if args.accuracy:
        metric_key = 'accuracies_by_thresh'    
    for summary in all_summary_files:
        summary_data = json.load(open(summary, 'r'))
        if metric_key not in summary_data:
            continue
        aucs_by_thresh = summary_data[metric_key]

        estimator = 'poselib'
        if 'opencv' in summary:
            estimator = 'opencv'

        #make sure everything is float
        for thresh in aucs_by_thresh:
            for k in aucs_by_thresh[thresh]:
                if isinstance(aucs_by_thresh[thresh][k], str):
                    aucs_by_thresh[thresh][k] = float(aucs_by_thresh[thresh][k].replace(' ', ''))

        # find best threshold based on the 5, 10, 20 mAP and everything is float
        df = pd.DataFrame(aucs_by_thresh).T.astype(float)
        df['mean'] = df.mean(axis=1)
        # create a string column called estimator
        cols = df.columns.tolist()
        dfs.append(df)
        names.append(summary_data['name'])
        estimators.append(estimator)

        # print(summary_data['name'])
        # print(df)
        # print()

    # use each col as the main col to determine the best threshold
    # for col in cols:
    col = 'mean'

    final_df = pd.DataFrame()
    # add cols
    final_df['name'] = names
    final_df['best_thresh'] = ''
    final_df['estimator'] = estimators
    final_df[cols] = -1.0

    for df, name, estimator in zip(dfs, names, estimators):
        best_thresh = df[col].idxmax()
        best_results = df.loc[best_thresh]

        # final_df.loc[final_df['name'] == name, 'best_thresh'] = best_thresh
        # final_df.loc[final_df['name'] == name and final_df['estimator'] == estimator, 'best_thresh'] = best_thresh
        # for _col in cols:
            # final_df.loc[final_df['name'] == name, _col] = best_results[_col]
        
        # now update the best_thresh based on the estimator
        final_df.loc[(final_df['name'] == name) & (final_df['estimator'] == estimator), 'best_thresh'] = best_thresh
        for _col in cols:
            final_df.loc[(final_df['name'] == name) & (final_df['estimator'] == estimator), _col] = best_results[_col]

    # sort by mean
    final_df = final_df.sort_values(by=['mean'])
    # reset index
    final_df = final_df.reset_index(drop=True)

    # drop estimator column
    final_df = final_df.drop(columns=['estimator'])

    # set max float precision to 1
    final_df = final_df.round(1)

    print(f"Dataset: {dataset_name}")
    print(f"Sorting by {col}")
    print(final_df)
    print()

    final_df.to_csv(os.path.join(FOLDER_RESULTS, f"{dataset_name}_{col}.csv"), index=False)

