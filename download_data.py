#!/usr/bin/env python3
"""Download data from HuggingFace for UNHRDB Students Edition."""

from huggingface_hub import hf_hub_download
import pandas as pd
import json
import os

def main():
    repo_id = 'lszoszk/treaty-bodies-general-comments'
    filename = 'data/train-00000-of-00001.parquet'
    
    print('Downloading data from HuggingFace...')
    local_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type='dataset')
    
    df = pd.read_parquet(local_path)
    print(f'Downloaded {len(df)} rows')
    print(f'Columns: {df.columns.tolist()}')
    
    # Save as JSON for easier frontend access
    output_dir = 'data'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save full dataset as JSON
    df.to_json(f'{output_dir}/general_comments.json', orient='records', indent=2)
    print(f'Saved {output_dir}/general_comments.json')
    
    # Save metadata
    metadata = {
        'total_documents': len(df),
        'columns': df.columns.tolist(),
        'source': f'https://huggingface.co/datasets/{repo_id}'
    }
    with open(f'{output_dir}/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print('Done!')

if __name__ == '__main__':
    main()