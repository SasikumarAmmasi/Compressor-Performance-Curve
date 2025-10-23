import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import xlsxwriter
import streamlit as st
from io import BytesIO
from matplotlib.patches import Patch

# ----------------------------------------------------------------------
# PLOTTING FUNCTIONS
# ----------------------------------------------------------------------

# PLOT 1: Qr2 vs. Discharge Pressure (Grouped by Suction Temperature)
def plot_qr2_vs_discharge_pressure_by_temp(df, df_sorted, pressure_value):
    """
    Generates the plot of Qr2 vs. Discharge Pressure, grouped by Suction Temperature.
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # --- A. PRIMARY X-AXIS (ax1, Bottom): Qr^2 (Reduced Flow) ---
    ax1.set_xlabel(r'Reduced Flow ($\mathbf{Qr^2}$)', fontsize=14)
    ax1.set_ylabel('Discharge Pressure ($\mathbf{barg}$)', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.7)

    grouped = df.groupby('Suction Temperature Deg C')
    unique_temps = sorted(df['Suction Temperature Deg C'].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_temps)))

    for i, temp in enumerate(unique_temps):
        group = grouped.get_group(temp).sort_values(by='Qr2')
        ax1.plot(
            group['Qr2'],
            group['Discharge Pressure barg'],
            linestyle='-',
            color=colors[i],
            label=f'{temp}°C'
        )
    
    # -----------------------------------------------------------
    # --- B. SECONDARY X-AXIS (ax3, Top): Actual Gas Flow (FIX) ---
    # -----------------------------------------------------------
    ax3 = ax1.twiny()

    fig.canvas.draw()
    major_qr2_ticks = ax1.get_xticks()

    qr2_values = df_sorted['Qr2'].values
    flow_values = df_sorted['Actual Gas Flow (Am3/hr)'].values
    
    flow_labels_amch = np.interp(
        major_qr2_ticks,
        qr2_values,
        flow_values
    ).astype(int)

    ax3.set_xticks(major_qr2_ticks)
    ax3.set_xticklabels(flow_labels_amch)

    flow_col = 'Actual Gas Flow (Am3/hr)'
    ax3.set_xlabel(r'Actual Gas Flow ($\mathbf{Am^3/hr}$)', fontsize=14, color='darkorange')
    ax3.tick_params(axis='x', labelcolor='darkorange', labelsize=10)
    ax3.set_xlim(ax1.get_xlim())

    ax1.set_title(f'Process Curve - Suction Pressure: {pressure_value} barg', fontsize=18)
    ax1.legend(title='Suction Temperature', loc='upper right')
    fig.tight_layout()
    
    plot_buffer = BytesIO()
    plt.savefig(plot_buffer, format='png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    plot_buffer.seek(0)
    
    plot_filename = f'Process_Curve_P_{pressure_value}.png'
    return plot_filename, plot_buffer


# PLOT 2: Complex Superimposed Map (Triple-Axis) - UPDATED SHADING LOGIC
def plot_superimposed_map_triple_axis(df, df_sorted, rated_power, pressure_value):
    """
    Generates the final superimposed plot with updated shading logic.
    Operating Zone (Green): ONLY where BOTH (Hr < Surge HR) AND (Power < Rated Power)
    Non-Operating Zone (Red): Anywhere else (above surge OR above rated power)
    """
    fig, ax1 = plt.subplots(figsize=(14, 8))
    qr2_for_shading = df_sorted['Qr2']

    # --- A. PRIMARY X-AXIS (ax1, Bottom): Qr^2 (Reduced Flow) ---
    ax1.set_xlabel(r'Reduced Flow Rate ($\mathbf{Qr^2}$)', fontsize=14)
    ax1.set_ylabel(r'Reduced Head ($\mathbf{Hr}$)', fontsize=14, color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Calculate Y1 (Hr) limits
    min_hr = min(df_sorted['Surge HR'].min(), df['Actual Hr'].min())
    max_hr = df['Actual Hr'].max()
    hr_range = max_hr - min_hr
    
    y1_min = max(0, min_hr - hr_range * 0.05)
    y1_max = max_hr + hr_range * 0.05
    ax1.set_ylim(y1_min, y1_max)

    # --- B. SECONDARY Y-AXIS (ax2, Right): Power (kW) ---
    ax2 = ax1.twinx()
    ax2.set_ylabel('Power (kW)', fontsize=14, color='g')
    ax2.tick_params(axis='y', labelcolor='g')

    # Calculate Y2 (Power) limits
    min_power = df['Power (kW)'].min()
    max_power = max(df['Power (kW)'].max(), rated_power)
    power_range = max_power - min_power
    
    y2_min = max(0, min_power - power_range * 0.05)
    y2_max = max_power + power_range * 0.05
    ax2.set_ylim(y2_min, y2_max)

    # --- CORRECTED SHADING LOGIC ---
    rated_power_array = np.full_like(qr2_for_shading, rated_power, dtype=float)
    surge_hr_array = df_sorted['Surge HR'].values
    power_array = df_sorted['Power (kW)'].values
    
    # Convert Power axis values to Hr axis coordinates for proper comparison
    # This allows us to compare surge line and rated power line on the same scale
    def power_to_hr_scale(power_val):
        """Convert power axis value to hr axis coordinate"""
        # Normalize power to [0,1] range on ax2
        power_norm = (power_val - y2_min) / (y2_max - y2_min)
        # Convert to hr scale
        hr_equiv = y1_min + power_norm * (y1_max - y1_min)
        return hr_equiv
    
    # Convert rated power line to Hr scale
    rated_power_in_hr_scale = power_to_hr_scale(rated_power)
    rated_power_hr_array = np.full_like(qr2_for_shading, rated_power_in_hr_scale, dtype=float)
    
    # Convert actual power values to Hr scale
    power_in_hr_scale = power_to_hr_scale(power_array)
    
    # Find the effective upper boundary (minimum of surge line and rated power in hr scale)
    # This represents the "safe" operating envelope
    effective_upper_boundary = np.minimum(surge_hr_array, rated_power_hr_array)
    
    # --------------------------------------------------------------------------
    # SHADING STRATEGY: 
    # 1. First fill EVERYTHING with RED (non-operating zone base)
    # 2. Then overlay GREEN only where it's safe (operating zone)
    # --------------------------------------------------------------------------
    
    # --------------------------------------------------------------------------
    # STEP 1: SHADE ENTIRE AREA AS NON-OPERATING ZONE (RED BASE LAYER)
    # --------------------------------------------------------------------------
    ax1.fill_between(
        qr2_for_shading, 
        y1_min, 
        y1_max,
        facecolor='red', 
        alpha=0.0, 
        zorder=1,
        interpolate=True
    )
    
    # --------------------------------------------------------------------------
    # STEP 2: SHADE OPERATING ZONE (GREEN) - OVERLAY ON TOP
    # Only where BOTH: below surge line AND below rated power line
    # This will cover the red underneath in the safe operating region
    # --------------------------------------------------------------------------
    ax1.fill_between(
        qr2_for_shading, 
        y1_min, 
        effective_upper_boundary,
        facecolor='green', 
        alpha=0.35, 
        zorder=2,
        interpolate=True
    )

    # --- CURVES (Draw last with highest zorder=3) ---

    # Plot 1: Surge HR vs. Qr^2 
    surge_line, = ax1.plot(
        df_sorted['Qr2'],
        df_sorted['Surge HR'],
        marker=None, 
        linestyle='-',
        color='red',
        linewidth=3.0,
        label=r'Surge Line ($\mathbf{Surge~Hr}$)',
        zorder=3
    )

    # Group by Temperature
    grouped = df.groupby('Suction Temperature Deg C')
    unique_temps = sorted(df['Suction Temperature Deg C'].unique())
    temp_colors = plt.cm.cool(np.linspace(0, 0.9, len(unique_temps)))
    actual_hr_handles = []
    for i, temp in enumerate(unique_temps):
        group = grouped.get_group(temp).sort_values(by='Qr2')
        line, = ax1.plot(
            group['Qr2'],
            group['Actual Hr'],
            marker=None,
            linestyle='-',
            color=temp_colors[i],
            linewidth=2.0,
            label=f'Hr @ {temp}°C',
            zorder=3
        )
        actual_hr_handles.append(line)

    # Plot 3: Power vs. Qr^2 (Grouped by Temp)
    power_handles = []
    for i, temp in enumerate(unique_temps):
        group = grouped.get_group(temp).sort_values(by='Qr2')
        line, = ax2.plot(
            group['Qr2'],
            group['Power (kW)'],
            marker=None,
            linestyle='--',
            color=temp_colors[i],
            linewidth=2.0,
            alpha=0.9,
            label=f'Pwr @ {temp}°C',
            zorder=3
        )
        power_handles.append(line)

    # Plot 4: Rated Power Line
    rated_power_line, = ax2.plot(
        df_sorted['Qr2'],
        [rated_power] * len(df_sorted),
        color='black',
        linestyle='-.',
        linewidth=2.0,
        label=f'Rated Power ({rated_power} kW)',
        zorder=3
    )

    # --- C. SECONDARY X-AXIS (ax3, Top): Actual Gas Flow ---
    ax3 = ax1.twiny()
    
    fig.canvas.draw()
    major_qr2_ticks = ax1.get_xticks()

    qr2_values = df_sorted['Qr2'].values
    flow_values = df_sorted['Actual Gas Flow (Am3/hr)'].values
    
    flow_labels_amch = np.interp(
        major_qr2_ticks,
        qr2_values,
        flow_values
    ).astype(int)

    ax3.set_xticks(major_qr2_ticks)
    ax3.set_xticklabels(flow_labels_amch)

    flow_col = 'Actual Gas Flow (Am3/hr)' 
    ax3.set_xlabel(r'Actual Gas Flow ($\mathbf{Am^3/hr}$)', fontsize=14, color='darkorange')
    ax3.tick_params(axis='x', labelcolor='darkorange', labelsize=10)
    ax3.set_xlim(ax1.get_xlim())

    # --- Legend Construction ---
    ax1.set_title(f'Compressor Performance Map - Suction Pressure: {pressure_value} barg', fontsize=18)

    # Proxy artists for the shading zones 
    hr_operating_zone_patch = Patch(facecolor='green', alpha=0.2, label='Operating Zone (Below Surge & Below Rated Power)')
    power_non_operating_zone_patch = Patch(facecolor='red', alpha=0.3, label='Non-Operating Zone (Above Surge OR Above Rated Power)')

    hr_legend_handles = [surge_line] + actual_hr_handles
    power_legend_handles = [rated_power_line] + power_handles

    all_handles = hr_legend_handles + power_legend_handles + [hr_operating_zone_patch, power_non_operating_zone_patch]
    all_labels = [h.get_label() for h in hr_legend_handles] + \
                 [h.get_label() for h in power_legend_handles] + \
                 [hr_operating_zone_patch.get_label(), power_non_operating_zone_patch.get_label()]

    ax1.legend(all_handles, all_labels,
               title='Curves & Zones (Hr Left, Pwr Right)',
               loc='upper left',
               bbox_to_anchor=(1.05, 1.0),
               ncol=1,
               fontsize=8)

    fig.tight_layout()

    plot_buffer = BytesIO()
    plt.savefig(plot_buffer, format='png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    plot_buffer.seek(0)
    
    plot_filename = f'Performance_Map_P_{pressure_value}.png'
    return plot_filename, plot_buffer

# ----------------------------------------------------------------------
# STREAMLIT MAIN EXECUTION SCRIPT
# ----------------------------------------------------------------------

def execute_plotting_and_excel_embedding():
    """
    Manages the Streamlit workflow: file upload, data analysis, plotting,
    and generating the output Excel file with embedded plots.
    """
    # Streamlit configuration
    st.set_page_config(layout="wide", page_title="Compressor Analysis Tool")
    st.title("🗜️ Compressor Performance Analysis and Reporting")
    st.markdown("Upload an Excel file containing compressor performance data to generate process curves and performance maps, grouped by unique suction pressure values.")
    
    # --- RATED POWER INPUT ---
    st.subheader("1. Configuration")
    rated_power = st.number_input(
        "Enter Compressor Rated Power (kW):",
        min_value=1,
        value=4481, # Default value
        step=10,
        help="This value defines the horizontal 'Rated Power' line on the Performance Map."
    )

    # --- Streamlit File Uploader ---
    st.subheader("2. Data Upload")
    uploaded_file = st.file_uploader(
        "Upload your Excel data file (.xlsx)", 
        type=['xlsx'],
        help="The file must contain the required columns for analysis."
    )
    
    if uploaded_file is None:
        st.info("Awaiting file upload...")
        return

    st.success(f"File uploaded: {uploaded_file.name}")
    
    # --- Data Reading and Validation ---
    try:
        uploaded_file.seek(0)
        df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"❌ Error reading Excel file. Error: {e}")
        return

    # Data Cleaning and Validation
    df.rename(columns={'Actual Gas Flow (AMCH)': 'Actual Gas Flow (Am3/hr)'}, inplace=True)

    required_columns = [
        'Suction Pressure barg', 'Suction Temperature Deg C', 
        'Discharge Pressure barg', 'Actual Gas Flow (Am3/hr)', 
        'Power (kW)', 'Actual Hr', 'Qr2', 'Surge HR'
    ]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        st.error(f"❌ Error: Missing required columns in the uploaded data: {missing_columns}")
        st.code(f"Required columns: {required_columns}", language="text")
        return

    # Analysis
    unique_pressures = sorted(df['Suction Pressure barg'].unique())
    st.subheader("3. Analysis & Plot Generation")
    st.info(f"✅ Found {len(unique_pressures)} unique suction pressures: {unique_pressures}")
    
    all_plot_data = []
    excel_buffer = BytesIO()
    
    # --- Create Excel Writer (in memory) ---
    try:
        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
            
            # 1. Write the Raw Data
            df.to_excel(writer, sheet_name='Raw Data Input', index=False)
            
            workbook = writer.book

            # Loop Through Each Suction Pressure and Generate Plots
            plot_progress = st.progress(0, text="Generating plots...")
            
            for i, pressure in enumerate(unique_pressures):
                df_pressure = df[df['Suction Pressure barg'] == pressure].copy()
                df_sorted_pressure = df_pressure.sort_values(by='Qr2').reset_index(drop=True)
                
                # Plot Set 1: Process Curve
                process_name, process_buffer = plot_qr2_vs_discharge_pressure_by_temp(df_pressure, df_sorted_pressure, pressure)
                all_plot_data.append((process_name, process_buffer))
                
                # Plot Set 2: Performance Map 
                performance_name, performance_buffer = plot_superimposed_map_triple_axis(df_pressure, df_sorted_pressure, rated_power, pressure)
                all_plot_data.append((performance_name, performance_buffer))
                
                # Update progress
                plot_progress.progress((i + 1) / len(unique_pressures), text=f"Generated plots for P: {pressure} barg")

            plot_progress.empty()
            st.success("All plots generated successfully!")
            
            # 2. Embed all plots
            for plot_name, plot_buffer in all_plot_data:
                sheet_name = plot_name.replace('.png', '').replace('Process_', 'Curve_')[:31] 
                worksheet = workbook.add_worksheet(sheet_name) 
                
                worksheet.insert_image('A1', plot_name, {
                    'image_data': plot_buffer, 
                    'x_scale': 0.7, 
                    'y_scale': 0.7
                })
                
            writer.close()
            excel_buffer.seek(0)
            
            # --- Streamlit Download Button ---
            output_filename = f'Compressor_Performance_Output_{os.path.splitext(uploaded_file.name)[0]}.xlsx'
            st.subheader("4. Download Results")
            st.download_button(
                label="⬇️ Download Analysis Excel File",
                data=excel_buffer,
                file_name=output_filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                help="Click to download the Excel file containing the raw data and all embedded plots."
            )
            
    except Exception as e:
        st.error(f"An unexpected error occurred during Excel processing or plotting: {e}")


# --- Execute the Streamlit app ---
if __name__ == '__main__':
    execute_plotting_and_excel_embedding()
