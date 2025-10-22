import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import xlsxwriter
import streamlit as st
from io import BytesIO

# ----------------------------------------------------------------------
# GLOBAL CONFIGURATION
# ----------------------------------------------------------------------
TEMP_DIR = 'temp_plots'


# ----------------------------------------------------------------------
# PLOTTING FUNCTIONS (PLOT 1 REMAINS SAME)
# ----------------------------------------------------------------------

# PLOT 1: Qr2 vs. Discharge Pressure (Grouped by Suction Temperature)
def plot_qr2_vs_discharge_pressure_by_temp(df, df_sorted, pressure_value):
    """
    Generates the plot of Qr2 vs. Discharge Pressure, grouped by Suction Temperature.
    MODIFIED: Notation for Reduced Flow is changed to Qr^2 in the X-axis label.
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

    # 1. Force the plotting to determine the automatic major ticks
    fig.canvas.draw()
    major_qr2_ticks = ax1.get_xticks()

    # 2. Get the full Qr2 range and corresponding Actual Gas Flow values
    qr2_values = df_sorted['Qr2'].values
    flow_values = df_sorted['Actual Gas Flow (Am3/hr)'].values
    
    # 3. Interpolate the Actual Gas Flow values for each major Qr2 tick position
    flow_labels_amch = np.interp(
        major_qr2_ticks,
        qr2_values,
        flow_values
    ).astype(int)

    # 4. Apply the new ticks and labels
    ax3.set_xticks(major_qr2_ticks)
    ax3.set_xticklabels(flow_labels_amch)

    flow_col = 'Actual Gas Flow (Am3/hr)' # For label consistency
    ax3.set_xlabel(r'Actual Gas Flow ($\mathbf{Am^3/hr}$)', fontsize=14, color='darkorange')
    ax3.tick_params(axis='x', labelcolor='darkorange', labelsize=10)
    ax3.set_xlim(ax1.get_xlim()) # Ensure the top axis limits match the bottom axis

    ax1.set_title(f'Process Curve - Suction Pressure: {pressure_value} barg', fontsize=18)
    ax1.legend(title='Suction Temperature', loc='upper right')
    fig.tight_layout()
    
    # Save plot to an in-memory buffer
    plot_buffer = BytesIO()
    plt.savefig(plot_buffer, format='png', dpi=300, bbox_inches='tight')
    plt.close(fig) # Close figure to free memory
    plot_buffer.seek(0)
    
    # Return a descriptive filename and the buffer
    plot_filename = f'Process_Curve_P_{pressure_value}.png'
    return plot_filename, plot_buffer


# PLOT 2: Complex Superimposed Map (Triple-Axis) - MODIFIED FOR SHADING
def plot_superimposed_map_triple_axis(df, df_sorted, rated_power, pressure_value):
    """
    Generates the final superimposed plot with Hr (Primary Y), Power (Secondary Y),
    and Actual Gas Flow (Secondary X).
    MODIFIED: Added green/red shading zones for Surge HR and Rated Power.
    """
    fig, ax1 = plt.subplots(figsize=(14, 8))

    # --- A. PRIMARY X-AXIS (ax1, Bottom): Qr^2 (Reduced Flow) ---
    ax1.set_xlabel(r'Reduced Flow Rate ($\mathbf{Qr^2}$)', fontsize=14)
    ax1.set_ylabel(r'Reduced Head ($\mathbf{Hr}$)', fontsize=14, color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Plot 1: Surge HR vs. Qr^2 (plain curve)
    surge_line, = ax1.plot(
        df_sorted['Qr2'],
        df_sorted['Surge HR'],
        marker=None, 
        linestyle='-',
        color='red',
        linewidth=3.0,
        label=r'Surge Line ($\mathbf{Surge~Hr}$)'
    )

    # Group by Temperature
    grouped = df.groupby('Suction Temperature Deg C')
    unique_temps = sorted(df['Suction Temperature Deg C'].unique())
    temp_colors = plt.cm.cool(np.linspace(0, 0.9, len(unique_temps)))

    # Plot 2: Actual Hr vs. Qr^2 (Grouped by Temp)
    actual_hr_handles = []
    for i, temp in enumerate(unique_temps):
        group = grouped.get_group(temp).sort_values(by='Qr2')
        line, = ax1.plot(
            group['Qr2'],
            group['Actual Hr'],
            marker=None,
            linestyle='-',
            color=temp_colors[i],
            linewidth=1.5,
            label=f'Hr @ {temp}°C'
        )
        actual_hr_handles.append(line)

    # --------------------------------------------------------------------------
    # --- ADD SHADING FOR SURGE LINE (Qr^2 vs Hr) ---
    # --------------------------------------------------------------------------
    min_hr, max_hr = ax1.get_ylim() # Get current Y-limits to define red zone top
    qr2_for_shading = df_sorted['Qr2']
    surge_hr_for_shading = df_sorted['Surge HR']

    # Green Zone (Below Surge HR)
    ax1.fill_between(qr2_for_shading, surge_hr_for_shading, min_hr, 
                     where=(surge_hr_for_shading >= min_hr),
                     facecolor='#90EE90', alpha=0.3, label='Safe Zone (Hr)') # Light Green

    # Red Zone (Above Surge HR)
    ax1.fill_between(qr2_for_shading, surge_hr_for_shading, max_hr,
                     where=(max_hr >= surge_hr_for_shading),
                     facecolor='#FFCCCB', alpha=0.3, label='Surge Zone (Hr)') # Light Red
    # --------------------------------------------------------------------------

    # --- B. SECONDARY Y-AXIS (ax2, Right): Power (kW) ---
    ax2 = ax1.twinx()
    ax2.set_ylabel('Power (kW)', fontsize=14, color='g')
    ax2.tick_params(axis='y', labelcolor='g')

    power_handles = []

    # Plot 3: Power vs. Qr^2 (Grouped by Temp)
    for i, temp in enumerate(unique_temps):
        group = grouped.get_group(temp).sort_values(by='Qr2')
        line, = ax2.plot(
            group['Qr2'],
            group['Power (kW)'],
            marker=None,
            linestyle='--',
            color=temp_colors[i],
            linewidth=1.5,
            alpha=0.6,
            label=f'Pwr @ {temp}°C'
        )
        power_handles.append(line)

    # Plot 4: Rated Power Line
    rated_power_line, = ax2.plot(
        df_sorted['Qr2'],
        [rated_power] * len(df_sorted),
        color='black',
        linestyle='-.',
        linewidth=2.0,
        label=f'Rated Power ({rated_power} kW)'
    )

    # --------------------------------------------------------------------------
    # --- ADD SHADING FOR RATED POWER LINE (Qr^2 vs Power) ---
    # --------------------------------------------------------------------------
    min_power, max_power = ax2.get_ylim() # Get current Y-limits for power axis
    rated_power_line_values = [rated_power] * len(df_sorted)

    # Green Zone (Below Rated Power)
    ax2.fill_between(qr2_for_shading, rated_power_line_values, min_power,
                     where=(rated_power_line_values >= min_power),
                     facecolor='#90EE90', alpha=0.3, label='Safe Zone (Pwr)') # Light Green

    # Red Zone (Above Rated Power)
    ax2.fill_between(qr2_for_shading, rated_power_line_values, max_power,
                     where=(max_power >= rated_power_line_values),
                     facecolor='#FFCCCB', alpha=0.3, label='Overload Zone (Pwr)') # Light Red
    # --------------------------------------------------------------------------

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

    # IMPORTANT: Include the fill_between legends here.
    # We need to collect handles from all axes and the fill_between calls.
    # Create proxy artists for the fill_between legends if they don't appear automatically
    from matplotlib.patches import Patch
    surge_safe_patch = Patch(facecolor='#90EE90', alpha=0.3, label='Hr Safe Zone')
    surge_red_patch = Patch(facecolor='#FFCCCB', alpha=0.3, label='Hr Surge Zone')
    power_safe_patch = Patch(facecolor='#90EE90', alpha=0.3, label='Pwr Safe Zone')
    power_red_patch = Patch(facecolor='#FFCCCB', alpha=0.3, label='Pwr Overload Zone')

    # Get handles for lines
    hr_legend_handles = [surge_line] + actual_hr_handles
    power_legend_handles = [rated_power_line] + power_handles

    all_handles = hr_legend_handles + power_legend_handles + [surge_safe_patch, surge_red_patch, power_safe_patch, power_red_patch]
    all_labels = [h.get_label() for h in hr_legend_handles] + \
                 [h.get_label() for h in power_legend_handles] + \
                 [surge_safe_patch.get_label(), surge_red_patch.get_label(), 
                  power_safe_patch.get_label(), power_red_patch.get_label()]


    # Use bbox_to_anchor for fine positioning
    ax1.legend(all_handles, all_labels,
               title='Curves & Zones (Hr Left, Pwr Right)',
               loc='upper left',
               bbox_to_anchor=(1.05, 1.0),
               ncol=1,
               fontsize=8)

    fig.tight_layout()

    # Save plot to an in-memory buffer
    plot_buffer = BytesIO()
    plt.savefig(plot_buffer, format='png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    plot_buffer.seek(0)
    
    # Return a descriptive filename and the buffer
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
    st.set_page_config(layout="wide", page_title="Compressor Analysis Tool")
    st.title("🗜️ Compressor Performance Analysis and Reporting")
    st.markdown("Upload an Excel file containing compressor performance data to generate process curves and performance maps, grouped by unique suction pressure values.")
    
    # --- RATED POWER INPUT ---
    st.subheader("1. Configuration")
    rated_power = st.number_input(
        "Enter Compressor Rated Power (kW):",
        min_value=1,
        value=4481, # Default value from the original code
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
        # Read the uploaded Excel file from the Streamlit buffer
        uploaded_file.seek(0) # Ensure pointer is at the start
        df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"❌ Error reading Excel file. Ensure it's a valid .xlsx file. Error: {e}")
        st.warning("If you see a dependency error (like 'openpyxl'), make sure it's listed in your requirements.txt.")
        return

    # Data Cleaning and Validation
    # NOTE: 'Actual Gas Flow (AMCH)' must be present in the original Excel file!
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

            plot_progress.empty() # Clear the progress bar after completion
            st.success("All plots generated successfully!")
            
            # 2. Embed all plots
            for plot_name, plot_buffer in all_plot_data:
                worksheet = workbook.add_worksheet(plot_name.replace('.png', '').replace('Process_', 'Curve_')) # Max 31 chars
                # Insert the plot from the in-memory buffer
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
