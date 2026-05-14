#!/usr/bin/env python3
"""
Audio Recording CSV Plotter (overlay)
Supports both `receiver,time,amplitude` CSVs and
`dataset,index_0,index_1,value` CSVs exported from sim_outs.
This variant overlays all receivers in the same plot (single axes).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def load_audio_csv_generic(csv_file):
    # read without forcing dtypes so we can detect special rows like /Fs_f
    df = pd.read_csv(csv_file, dtype=str)

    sample_rate = None
    cols = [c.strip() for c in df.columns]

    # receiver,time,amplitude layout
    if set(['receiver', 'time', 'amplitude']).issubset(cols):
        layout = 'receiver_time_amp'
        # detect and extract the sample-rate marker row before dropping missing amplitudes
        receiver_text = df['receiver'].fillna('').astype(str).str.strip()
        time_numeric = pd.to_numeric(df['time'], errors='coerce')
        amplitude_numeric = pd.to_numeric(df['amplitude'], errors='coerce')
        fs_rows = df[
            (receiver_text == '')
            & time_numeric.notna()
            & (time_numeric > 1000)
            & amplitude_numeric.isna()
        ]
        if not fs_rows.empty:
            sample_rate = float(time_numeric.loc[fs_rows.index[0]])
            df = df.drop(index=fs_rows.index)

        # convert numeric cols for actual signal rows
        df['time'] = pd.to_numeric(df['time'], errors='coerce')
        df['amplitude'] = pd.to_numeric(df['amplitude'], errors='coerce')
        df = df.dropna(subset=['time', 'amplitude'])
        return {'layout': layout, 'df': df, 'sample_rate': sample_rate}

    # indexed layout (dataset, index_0, index_1, value)
    if set(['dataset', 'index_0', 'index_1', 'value']).issubset(cols):
        layout = 'indexed'
        # extract sample-rate if present in a dataset named /Fs_f or Fs_f
        # don't drop rows yet
        df_dataset = df['dataset'].astype(str)
        fs_rows = df[df_dataset.str.strip().isin(['/Fs_f', 'Fs_f'])]
        if not fs_rows.empty:
            try:
                sample_rate = float(fs_rows['value'].astype(float).iloc[0])
            except Exception:
                sample_rate = None

        # choose primary dataset to plot: prefer /r_out_f, then /r_out, else first non-Fs dataset
        primary_mask = df_dataset.str.strip().isin(['/r_out_f', '/r_out', 'r_out_f', 'r_out'])
        if primary_mask.any():
            primary_name = df_dataset[primary_mask].iloc[0]
        else:
            # pick first dataset that's not Fs
            non_fs = df_dataset[~df_dataset.str.strip().isin(['/Fs_f', 'Fs_f'])]
            primary_name = non_fs.iloc[0] if not non_fs.empty else df_dataset.iloc[0]

        # filter to primary dataset rows
        df_plot = df[df_dataset == primary_name].copy()
        # coerce numeric columns
        df_plot['index_0'] = pd.to_numeric(df_plot['index_0'], errors='coerce')
        df_plot['index_1'] = pd.to_numeric(df_plot['index_1'], errors='coerce')
        df_plot['value'] = pd.to_numeric(df_plot['value'], errors='coerce')
        df_plot = df_plot.dropna(subset=['index_0', 'index_1', 'value'])
        # attach dataset name for downstream plotting
        df_plot['dataset'] = df_plot['dataset'].astype(str)
        return {'layout': layout, 'df': df_plot, 'sample_rate': sample_rate}

    return {'layout': 'unknown', 'df': df, 'sample_rate': sample_rate}


def plot_sim_indexed(df, fs=None, indices=None, max_plots=8, output_file=None, figsize=(14,6)):
    # Pivot so rows=index_0, columns=index_1
    pivot = df.pivot_table(index='index_0', columns='index_1', values='value')
    pivot = pivot.sort_index()

    available_indices = list(pivot.index.astype(int))
    if indices is None:
        plot_indices = available_indices
    else:
        plot_indices = [int(i) for i in indices]

    time_idx = np.array(pivot.columns.astype(int))
    if fs:
        time = time_idx / float(fs)
        xlabel = 'Time (s)'
    else:
        time = time_idx
        xlabel = 'Sample'

    # Overlay all selected index_0 (receivers) on the same axes
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    cmap = plt.get_cmap('tab10')
    n_lines = len(plot_indices)
    for i, idx in enumerate(plot_indices):
        if idx not in pivot.index:
            continue
        row = pivot.loc[idx].values
        color = cmap(i % 10) if n_lines <= 10 else None
        ax.plot(time, row, linewidth=0.8, label=f'R{int(idx)+1}', color=color)

    ax.set_xlabel(xlabel)
    ax.set_ylabel('Amplitude')
    ax.set_title('Receivers (overlaid)')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize='small')
    plt.tight_layout()
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
    else:
        plt.show()


def plot_receiver_time_overlay(df, fs=None, output_file=None, figsize=(14,6)):
    """
    Plot all receivers overlaid on the same axes with a legend.
    """
    receivers = df['receiver'].unique()
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    for r in receivers:
        d = df[df['receiver'] == r].sort_values('time')
        time = d['time'].astype(float).values
        amp = d['amplitude'].astype(float).values
        if fs:
            time = time / float(fs)
        ax.plot(time, amp, linewidth=0.8, label=str(r))

    if fs:
        ax.set_xlabel('Time (s)')
    else:
        ax.set_xlabel('Sample')

    ax.set_ylabel('Amplitude')
    ax.set_title('Receivers (overlaid)')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize='small')
    plt.tight_layout()
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='CSV audio plotter (overlay)')
    parser.add_argument('csv_file')
    parser.add_argument('-o','--output')
    parser.add_argument('--spectrogram', action='store_true')
    parser.add_argument('--fs', type=float, default=None, help='sample rate in Hz')
    parser.add_argument('--indices', help='comma-separated index_0 values to plot (for indexed CSV)')
    parser.add_argument('--max-plots', type=int, default=8)
    args = parser.parse_args()

    info = load_audio_csv_generic(args.csv_file)
    layout = info['layout']
    df = info['df']
    detected_fs = info['sample_rate']
    fs = args.fs if args.fs is not None else detected_fs

    if layout == 'receiver_time_amp':
        plot_receiver_time_overlay(df, fs=fs, output_file=args.output)
    elif layout == 'indexed':
        indices = None
        if args.indices:
            indices = args.indices.split(',')
        plot_sim_indexed(df, fs=fs, indices=indices, max_plots=args.max_plots, output_file=args.output)
    else:
        print('Unknown CSV layout. Columns found:', df.columns.tolist())

if __name__ == '__main__':
    main()
