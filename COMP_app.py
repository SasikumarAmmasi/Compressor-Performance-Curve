import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import xlsxwriter
import streamlit as st
from io import BytesIO
from matplotlib.patches import Patch
from scipy.interpolate import interp1d

# ----------------------------------------------------------------------
# GLOBAL CONFIGURATION
# ----------------------------------------------------------------------
TEMP_DIR = 'temp_plots'


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


# PLOT 2: Complex Superimposed Map (Triple-Axis) - MODIFIED FOR FILL_BETWEEN INTERSECTION
def plot_superimposed_map_triple_axis(df, df_sorted, rated_power, pressure_value):
    """
    Generates the final superimposed plot with Hr (Primary Y), Power (Secondary Y).
    MODIFIED: Shading is a single fill_between based on the MINIMUM of the Surge HR line 
    and the calculated Power Limit Hr curve.
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
    # --- NEW SHADING LOGIC: COMBINED MINIMUM BOUNDARY (FILL_BETWEEN) ---
    # --------------------------------------------------------------------------
    
    # Initialize the Power Limit Hr boundary to a very high value
    qr2_for_shading = df_sorted['Qr2'].values
    power_limit_hr = np.full_like(qr2_for_shading, np.nan)
    
    # 1. Calculate the Power Limit Hr boundary across Qr2
    for temp in unique_temps:
        group = grouped.get_group(temp).sort_values(by='Qr2')
        
        # Check if Power (kW) monotonically increases with Qr2 for proper interpolation
        # This is a key simplification needed for this approach.
        if group['Qr2'].is_monotonic_increasing and group['Power (kW)'].is_monotonic_increasing:
            # Interpolate Qr2 vs. Power (kW)
            power_interp_func = interp1d(group['Power (kW)'], group['Qr2'], kind='linear', fill_value="extrapolate")
            
            # Find the Qr2 corresponding to the Rated Power
            qr2_at_rated_power = power_interp_func(rated_power)
            
            # If Qr2 is within the data range, find the corresponding Hr value
            if group['Qr2'].min() <= qr2_at_rated_power <= group['Qr2'].max():
                hr_interp_func = interp1d(group['Qr2'], group['Actual Hr'], kind='linear')
                hr_at_rated_power = hr_interp_func(qr2_at_rated_power)
                
                # Create a power limit curve (Hr value) for this temperature across the entire Qr2 range.
                # The constraint is P < Prated. Since Power increases with Qr2, Qr2 must be < Qr2_at_rated_power.
                # The Hr boundary for the power constraint is effectively Hr_at_rated_power for Qr2 < Qr2_at_rated_power, 
                # and Hr_max for Qr2 > Qr2_at_rated_power (since it's not the constraint there).
                
                temp_hr_limit = np.full_like(qr2_for_shading, np.nan)
                
                # Find the index where Qr2 is less than the limit
                idx_below_limit = qr2_for_shading < qr2_at_rated_power
                
                # For Qr2 values below the power limit, the effective Hr constraint is Hr_at_rated_power
                # This is a conceptual simplification. The true safe zone must be the minimum of all Hr curves at the power limit.
                # Instead of simplifying, we take the MINIMUM Hr across ALL safe data points.

                # Simplified Approach: Use the lowest Hr where Power is equal to Rated Power.
                # For now, we'll track the lowest Hr value that *hits* rated power for proper boundary definition
                
                # This complex interpolation loop is the cleanest way to define the Power Limit Hr boundary
                # for a single Qr2-based curve. We create a "Power Limit Hr Boundary" curve.
                
                
                # We will define the Power Limit Hr curve by taking the lowest Hr value (max constraint)
                # at a given Qr2 from all temperature curves that remain below Rated Power.
                
                # Interpolate Qr2 vs Hr for current temp group
                hr_for_qr2_func = interp1d(group['Qr2'], group['Actual Hr'], kind='linear', bounds_error=False, fill_value=np.nan)
                
                # Create a temporary Hr constraint array: set Hr_at_rated_power to all Qr2 values
                # If we take the MINIMUM across all Power Limit Hr curves, the lowest Hr curve (highest constraint) wins.
                
                temp_power_limit_hr = np.full_like(qr2_for_shading, np.nan)
                
                # The power constraint is essentially a vertical cut-off. 
                # If Power(Qr2) > Prated, the Hr is forbidden.
                
                # A more correct approach is to define the "Power Limit Hr" curve as the MINIMUM Hr curve
                # (which is the hottest temp curve) *where that curve hits rated power*.
                
                # Let's find the Qr2 where the Hottest curve hits rated power (highest constraint)
                
                
                # Since the actual Hr curves are already plotted, let's simplify the definition
                # of the Power Limit Hr Boundary curve.
                
                # For each Qr2, the maximum safe Hr is the Hr of the curve that hits Prated *at that Qr2*.
                
                
                # For this specific implementation, we will define the Power Limit Hr by finding the 
                # lowest Hr associated with the Rated Power. This gives a single, restrictive boundary.
                
                
                # The following finds the Qr2 where the curve hits Prated, and sets the Hr value
                # for all Qr2 to the left of that point, and NaN (i.e., no constraint) to the right.
                
                if group['Qr2'].min() <= qr2_at_rated_power <= group['Qr2'].max():
                    # Create the constraint curve for this temperature:
                    # Hr = Hr_at_rated_power for Qr2 < Qr2_at_rated_power
                    # Hr = infinity for Qr2 >= Qr2_at_rated_power
                    
                    hr_at_rated_power_safe = hr_interp_func(qr2_at_rated_power)
                    
                    # Create a temporary constraint curve (Hr) for this temperature
                    temp_hr_constraint = np.full_like(qr2_for_shading, np.nan)
                    
                    # Find the interpolated Hr value for all Qr2 < Qr2_at_rated_power
                    idx_safe = qr2_for_shading < qr2_at_rated_power
                    
                    # For a given Qr2, the Hr constraint is defined by the Hr of the *actual* operating curve
                    # that hits rated power. This is complex.
                    
                    # The simplest and most restrictive power limit constraint is a vertical line at the 
                    # SMALLEST Qr2 where ANY curve exceeds rated power.
                    
                    # To be conservative, we should find the **minimum Qr2** across all temperatures that hits rated power.
                    if np.isnan(power_limit_hr).all():
                         power_limit_hr = np.full_like(qr2_for_shading, np.inf)

                    # For all Qr2 less than the current temperature's power limit Qr2, 
                    # the effective Hr constraint for the power limit is simply the highest Hr (safest) among all temps.
                    
                    # The MINIMUM Hr where Power < Prated for ALL Qr2 is what defines the boundary.
                    
                    
                    # Let's take the lowest Hr where power is rated across the range
                    
                    
                    # We will define the Power Limit Hr boundary as the maximum of all *actual* Hr curves
                    # that are safely below Rated Power. This is not smooth.
                    
                    # Let's simplify: Take the lowest Hr at any given Qr2 that is still below Rated Power.
                    # This results in the most restrictive safe boundary.
                    
                    
                    # Revert to the best simple approximation: Power limit is a vertical line.
                    
                    # We find the MINIMUM Qr2 where the Rated Power line is hit across ALL temperatures
                    
                    # Only the highest (most restrictive) Hr boundary must be considered.
                    
                    if np.isnan(power_limit_hr).all():
                        power_limit_hr = hr_for_qr2_func(qr2_for_shading)
                    else:
                        power_limit_hr = np.minimum(power_limit_hr, hr_for_qr2_func(qr2_for_shading))
                        
    # This logic is proving too complex for robust, generic interpolation across multiple curves.
    # The scatter plot was the correct robust solution for the intersection.
    
    # Let's revert to a simpler, visually effective, non-interpolated approximation:
    # We find the single Qr2 that represents the most flow-restrictive power limit.
    
    # 1. Find the Qr2 where Power = Prated for the *hottest* (most restrictive) Hr curve
    
    # To use fill_between and respect the intersection, we must define the Combined Limit Hr boundary.
    
    # We use a combined array to find the most restrictive Hr at every Qr2
    
    # Get all Qr2, Hr, Power values
    all_qr2 = df['Qr2'].values
    all_hr = df['Actual Hr'].values
    all_pwr = df['Power (kW)'].values
    
    # Filter points where Power is below Rated Power
    pwr_safe_mask = all_pwr <= rated_power
    
    # Map the power-safe Hr points to a restrictive boundary.
    # We take the maximum safe Hr observed at a given Qr2 from ALL safe points.
    
    # Create a DataFrame of safe points for easier manipulation
    df_safe = df[pwr_safe_mask].copy().sort_values(by='Qr2')
    
    # Create an interpolator for the upper Hr boundary defined by the Power limit.
    # We take the maximum Hr value for any given Qr2 where the power is still safe.
    # Since Hr is dependent on Temp (and Qr2), the highest Hr for a given Qr2 (safest Hr)
    # in the safe power zone defines the boundary.
    
    # Use aggregation to find the maximum safe Hr for smooth interpolation
    df_agg = df_safe.groupby(df_safe['Qr2']).agg({
        'Actual Hr': 'max'
    }).reset_index()
    
    # Create the interpolated Power Limit Hr curve (on ax1 scale)
    if len(df_agg) > 1:
        power_limit_hr_func = interp1d(df_agg['Qr2'], df_agg['Actual Hr'], kind='linear', bounds_error=False, fill_value=np.nan)
        power_limit_hr_boundary = power_limit_hr_func(qr2_for_shading)
    else:
        # If not enough safe points, assume no power constraint
        power_limit_hr_boundary = np.full_like(qr2_for_shading, np.inf) 
        
    # 2. Combine with the Surge Hr Line (the other boundary)
    combined_hr_limit = np.minimum(df_sorted['Surge HR'].values, power_limit_hr_boundary)

    # Clean up NaNs which will happen outside the observed data range
    combined_hr_limit[np.isnan(combined_hr_limit)] = np.inf 
    
    # 3. Apply fill_between using the combined boundary
    ax1.fill_between(
        qr2_for_shading, 
        combined_hr_limit, 
        ax1.get_ylim()[0],
        where=(combined_hr_limit > ax1.get_ylim()[0]), # Only fill where the limit is above the axis bottom
        facecolor='green', 
        alpha=0.15, 
        zorder=0,
        label='Combined Operating Zone'
    ) 
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

    # Create proxy artist for the shading legend
    operating_zone_patch = Patch(facecolor='green', alpha=0.15, label='Combined Operating Zone')

    # Get handles for lines
    hr_legend_handles = [surge_line] + actual_hr_handles
    power_legend_handles = [rated_power_line] + power_handles

    all_handles = hr_legend_handles + power_legend_handles + [operating_zone_patch]
    all_labels = [h.get_label() for h in hr_legend_handles] + \
                 [h.get_label() for h in power_legend_handles] + \
                 [operating_zone_patch.get_label()]


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
# STREAMLIT MAIN EXECUTION SCRIPT (REMAINS UNTOUCHED)
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
