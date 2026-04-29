"""
Script to visualize the computational and carbon footprint of IDUN cluster jobs.
It fetches SLURM job history (via sacct), filters for GPU jobs, calculates 
the duration and estimated CO2 emissions, and generates a dual-axis bar 
chart comparing execution time and carbon footprint.
"""

import matplotlib.pyplot as plt
import pandas as pd
import subprocess
import numpy as np

cmd = "sacct -S 2026-03-01 -u $USER --format=JobID,JobName,Elapsed,AllocTRES --noheader --parsable2"
data = subprocess.check_output(cmd, shell=True).decode('utf-8')

jobs = []
for line in data.strip().split('\n'):
    parts = line.split('|')
    if len(parts) >= 4 and 'gpu' in parts[3] and "." not in parts[0]:
        try:
            h, m, s = map(int, parts[2].split(':')) if ':' in parts[2] else (0,0,0)
            duration = h * 60 + m + s / 60
            jobs.append({'ID': parts[0], 'Duration': duration})
        except: continue

df = pd.DataFrame(jobs)

# GROUPING LOGIC: Filter jobs lasting more than 2 hours; aggregate shorter jobs
main_jobs = df[df['Duration'] >= 120].copy()
small_jobs_sum = df[df['Duration'] < 120]['Duration'].sum()

if small_jobs_sum > 0:
    new_row = pd.DataFrame([{'ID': 'Short Jobs (<2h)', 'Duration': small_jobs_sum}])
    df_plot = pd.concat([main_jobs, new_row], ignore_index=True)
else:
    df_plot = main_jobs

df_plot['CO2'] = (df_plot['Duration'] / 60) * 0.25 * 30

# Plotting
fig, ax1 = plt.subplots(figsize=(10, 6))
x = np.arange(len(df_plot))
width = 0.35

b1 = ax1.bar(x - width/2, df_plot['Duration'], width, color='forestgreen', alpha=0.7, label='Time (min)')
ax2 = ax1.twinx()
b2 = ax2.bar(x + width/2, df_plot['CO2'], width, color='firebrick', alpha=0.7, label='Carbon (gCO2)')

for i in range(len(df_plot)):
    duration = df_plot.iloc[i]['Duration']
    co2 = df_plot.iloc[i]['CO2']
    
    # Calculate the X position to align text to the left edge of the green bar
    # (x[i] is the center, width/2 is the bar offset, so x[i] - width is the left edge)
    x_pos = x[i] - width 

    # 1. Time (Green text) - Displayed just above the bar
    ax1.text(x_pos, duration + (df_plot['Duration'].max() * 0.02), 
             f"{duration:.0f}m", color='darkgreen', 
             fontsize=9, fontweight='bold', ha='left')

    # 2. Carbon (Red text) - Displayed higher up, aligned to the same X position
    # Uses ax2 to ensure the text scales correctly with the CO2 axis
    ax2.text(x_pos, co2 + (df_plot['CO2'].max() * 0.08), 
             f"{co2:.1f}g", color='darkred', 
             fontsize=9, fontweight='bold', ha='left')

# --- Y-AXIS LIMIT ADJUSTMENTS TO PREVENT TEXT CLIPPING ---
ax1.set_ylim(0, df_plot['Duration'].max() * 1.30) 
ax2.set_ylim(0, df_plot['CO2'].max() * 1.30)

ax1.set_xticks(x)
ax1.set_xticklabels(df_plot['ID'], rotation=25, ha='right')
ax1.set_title('SER summary : Focus on the major phases', pad=30)
ax1.set_ylim(0, df_plot['Duration'].max() * 1.15) # Add top margin
ax2.set_ylim(0, df_plot['CO2'].max() * 1.15)

plt.tight_layout()
plt.savefig('Training_results/compute_footprint_clean.png')
