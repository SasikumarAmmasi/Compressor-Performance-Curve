import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import xlsxwriter
from google.colab import files
from io import StringIO # Retained for compatibility but not used for Excel upload

# ----------------------------------------------------------------------
# GLOBAL CONFIGURATION
# ----------------------------------------------------------------------
# This constant is now defined once and used throughout the script.
RATED_POWER = 4481
TEMP_DIR = 'temp_plots'
os.makedirs(TEMP_DIR, exist_ok=True)


# ----------------------------------------------------------------------
# PLOTTING FUNCTIONS (YOUR ORIGINAL LOGIC - MODIFIED FOR FILE SAVING)
# ----------------------------------------------------------------------

# PLOT 1: Qr2 vs. Discharge Pressure (Grouped by Suction Temperature)
# Added 'pressure_value' argument for the title and changed output from plt.show() to savefig/return filename.
def plot_qr2_vs_discharge_pressure_by_temp(df, df_sorted, pressure_value):
    """
    Generates and saves the plot of Qr2 vs. Discharge Pressure, grouped by Suction Temperature.
    """
    fig = plt.figure(figsize=(10, 6))
    ax = fig.gca()

    grouped = df.groupby('Suction Temperature Deg C')
    unique_temps = sorted(df['Suction Temperature Deg C'].unique())

    # Generate colors for the curves
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_temps)))

    for i, temp in enumerate(unique_temps):
        group = grouped.get_group(temp).sort_values(by='Qr2')
        ax.plot(
            group['Qr2'],
            group['Discharge Pressure barg'],
            marker='o',
            linestyle='-',
            color=colors[i],
            label=f'{temp}°C'
        )

    ax.set_title(f'Process Curve - Suction Pressure: {pressure_value} barg', fontsize=18)
    ax.set_xlabel(r'Reduced Flow ($\mathbf{Q_{r2}}$)', fontsize=14)
    ax.set_ylabel('Discharge Pressure ($\mathbf{barg}$)', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(title='Suction Temperature', loc='upper right')
    fig.tight_layout()
    
    # --- MODIFICATION: Save Plot ---
    plot_filename = os.path.join(TEMP_DIR, f'Process_Curve_P_{pressure_value}.png')
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close(fig) # Close figure to free memory
    return plot_filename


# PLOT 2: Complex Superimposed Map (Triple-Axis)
# Added 'pressure_value' argument for the title and changed output from plt.show() to savefig/return filename.
def plot_superimposed_map_triple_axis(df, df_sorted, rated_power, pressure_value):
    """
    Generates and saves the final superimposed plot with Hr (Primary Y), Power (Secondary Y),
    and Actual Gas Flow (Secondary X).
    """
    fig, ax1 = plt.subplots(figsize=(14, 8))

    # --- A. PRIMARY X-AXIS (ax1, Bottom): Qr2 (Reduced Flow) ---
    ax1.set_xlabel(r'Reduced Flow Rate ($\mathbf{Q_{r2}}$)', fontsize=14)
    ax1.set_ylabel(r'Reduced Head ($\mathbf{Hr}$)', fontsize=14, color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Plot 1: Surge HR vs. Qr2 (Uses the pre-sorted df_sorted)
    surge_line, = ax1.plot(
        df_sorted['Qr2'],
        df_sorted['Surge HR'],
        marker='s',
        linestyle='-',
        color='red',
        linewidth=3.0,
        label=r'Surge Line ($\mathbf{Surge~Hr}$)'
    )

    # Group by Temperature
    grouped = df.groupby('Suction Temperature Deg C')
    unique_temps = sorted(df['Suction Temperature Deg C'].unique())
    temp_colors = plt.cm.cool(np.linspace(0, 0.9, len(unique_temps)))

    # Plot 2: Actual Hr vs. Qr2 (Grouped by Temp)
    actual_hr_handles = []
    for i, temp in enumerate(unique_temps):
        group = grouped.get_group(temp).sort_values(by='Qr2')
        line, = ax1.plot(
            group['Qr2'],
            group['Actual Hr'],
            marker='o',
            linestyle='-',
            color=temp_colors[i],
            linewidth=1.5,
            markersize=5,
            label=f'Hr @ {temp}°C'
        )
        actual_hr_handles.append(line)

    # --- B. SECONDARY Y-AXIS (ax2, Right): Power (kW) ---
    ax2 = ax1.twinx()
    ax2.set_ylabel('Power (kW)', fontsize=14, color='g')
    ax2.tick_params(axis='y', labelcolor='g')

    power_handles = []

    # Plot 3: Power vs. Qr2 (Grouped by Temp)
    for i, temp in enumerate(unique_temps):
        group = grouped.get_group(temp).sort_values(by='Qr2')
        line, = ax2.plot(
            group['Qr2'],
            group['Power (kW)'],
            marker='^',
            linestyle='--',
            color=temp_colors[i],
            linewidth=1.5,
            markersize=5,
            alpha=0.6,
            label=f'Pwr @ {temp}°C'
        )
        power_handles.append(line)

    # Plot 4: Rated Power Line (Uses the pre-sorted df_sorted)
    rated_power_line, = ax2.plot(
        df_sorted['Qr2'],
        [rated_power] * len(df_sorted),
        color='black',
        linestyle='-.',
        linewidth=2.0,
        label=f'Rated Power ({rated_power} kW)'
    )

    # --- C. SECONDARY X-AXIS (ax3, Top): Actual Gas Flow ---
    ax3 = ax1.twiny()

    # Use the pre-sorted df_sorted for axis alignment
    n_labels = 6
    flow_indices = np.linspace(0, len(df_sorted) - 1, n_labels).astype(int)

    # Rename the column used for the axis label
    df_sorted.rename(columns={'Actual Gas Flow (AMCH)': 'Actual Gas Flow (Am3/hr)'}, inplace=True)
    
    flow_labels_qr2 = df_sorted['Qr2'].iloc[flow_indices].values
    flow_labels_amch = df_sorted['Actual Gas Flow (Am3/hr)'].iloc[flow_indices].values.astype(int)

    ax3.set_xticks(flow_labels_qr2)
    ax3.set_xticklabels(flow_labels_amch)

    ax3.set_xlabel(r'Actual Gas Flow ($\mathbf{Am^3/hr}$)', fontsize=14, color='darkorange')
    ax3.tick_params(axis='x', labelcolor='darkorange')
    ax3.set_xlim(ax1.get_xlim())

    # --- Legend Construction (Offset to prevent overlap) ---
    ax1.set_title(f'Compressor Performance Map - Suction Pressure: {pressure_value} barg', fontsize=18)

    # Combine all handles and labels for a single legend
    hr_legend_handles = [surge_line] + actual_hr_handles
    power_legend_handles = [rated_power_line] + power_handles

    all_handles = hr_legend_handles + power_legend_handles
    all_labels = [h.get_label() for h in all_handles]

    # Use bbox_to_anchor for fine positioning
    ax1.legend(all_handles, all_labels,
               title='Curves (Hr Left, Pwr Right, Temp-Colored)',
               loc='upper left',
               bbox_to_anchor=(1.05, 1.0),
               ncol=1,
               fontsize=8)

    fig.tight_layout()

    # --- MODIFICATION: Save Plot ---
    plot_filename = os.path.join(TEMP_DIR, f'Performance_Map_P_{pressure_value}.png')
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return plot_filename

# ----------------------------------------------------------------------
# MAIN EXECUTION SCRIPT (Handles Upload, Looping, and Excel Output)
# ----------------------------------------------------------------------

def execute_plotting_and_excel_embedding():
    """
    Manages the entire workflow: file upload, data analysis, plotting,
    and embedding plots into a new Excel file.
    """
    
    print("-------------------------------------------------------")
    print("1. DATA UPLOAD: Click 'Choose Files' to upload your Excel data file (.xlsx).")
    print("-------------------------------------------------------")
    
    uploaded = files.upload()
    
    if not uploaded:
        print("No file uploaded. Exiting.")
        return

    file_name = list(uploaded.keys())[0]
    
    try:
        # Read the uploaded Excel file
        df = pd.read_excel(file_name)
    except Exception as e:
        print(f"Error reading Excel file. Ensure it's a valid .xlsx file. Error: {e}")
        return

    # --- Data Cleaning and Validation ---
    
    # Rename the column to match the plotting function's usage
    df.rename(columns={'Actual Gas Flow (AMCH)': 'Actual Gas Flow (Am3/hr)'}, inplace=True)

    required_columns = ['Suction Pressure barg', 'Suction Temperature Deg C', 
                        'Discharge Pressure barg', 'Actual Gas Flow (Am3/hr)', 
                        'Power (kW)', 'Actual Hr', 'Qr2', 'Surge HR']
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        print(f"❌ Error: Missing required columns in the uploaded data: {missing_columns}")
        return

    # Analysis
    unique_pressures = sorted(df['Suction Pressure barg'].unique())
    print(f"\n✅ Found {len(unique_pressures)} unique suction pressures: {unique_pressures}")
    
    all_plot_files = []
    
    # Loop Through Each Suction Pressure and Generate Plots
    for pressure in unique_pressures:
        df_pressure = df[df['Suction Pressure barg'] == pressure].copy()
        
        # Sort the data for this pressure level, as required by your plotting functions
        df_sorted_pressure = df_pressure.sort_values(by='Qr2').reset_index(drop=True)
        
        print(f"\n--- Generating plots for Suction Pressure {pressure} barg ---")
        
        # Plot Set 1: Process Curve
        process_file = plot_qr2_vs_discharge_pressure_by_temp(df_pressure, df_sorted_pressure, pressure)
        all_plot_files.append((f'Curve_P{pressure}', process_file))
        
        # Plot Set 2: Performance Map
        performance_file = plot_superimposed_map_triple_axis(df_pressure, df_sorted_pressure, RATED_POWER, pressure)
        all_plot_files.append((f'Map_P{pressure}', performance_file))

    # --- Create Final Excel File with Data and Embedded Plots ---
    output_filename = f'Compressor_Performance_Output_{os.path.splitext(file_name)[0]}.xlsx'
    writer = pd.ExcelWriter(output_filename, engine='xlsxwriter')
    
    # 1. Write the Raw Data
    df.to_excel(writer, sheet_name='Raw Data Input', index=False)

    # 2. Embed all plots
    workbook = writer.book
    
    for plot_name, plot_path in all_plot_files:
        worksheet = workbook.add_worksheet(plot_name)
        # Scale down for better Excel viewing
        worksheet.insert_image('A1', plot_path, {'x_scale': 0.7, 'y_scale': 0.7}) 
        print(f"   -> Embedded {plot_name} into output Excel.")

    # Close the Excel writer
    writer.close()
    
    # Clean up temporary plot images
    for _, plot_path in all_plot_files:
        os.remove(plot_path)
    os.rmdir(TEMP_DIR)
    
    print(f"\n-------------------------------------------------------")
    print(f"2. DOWNLOAD: Successfully created output file: {output_filename}")
    print("-------------------------------------------------------")
    
    # Offer the final file for download
    files.download(output_filename)
    

# --- Execute the main function ---
if __name__ == '__main__':
    execute_plotting_and_excel_embedding()
