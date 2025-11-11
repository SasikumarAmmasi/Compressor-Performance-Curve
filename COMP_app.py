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
    fig, ax1 = plt.subplots(figsize=(13.5, 8))

    # --- A. PRIMARY X-AXIS (ax1, Bottom): Qr^2 (Reduced Flow) ---
    ax1.set_xlabel(r'Reduced Flow ($\mathbf{Qr^2}$)', fontsize=14)
    ax1.set_ylabel(r'Discharge Pressure ($\mathbf{barg}$)', fontsize=14)
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
    fig, ax1 = plt.subplots(figsize=(14, 10))
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
    def power_to_hr_scale(power_val):
        """Convert power axis value to hr axis coordinate"""
        power_norm = (power_val - y2_min) / (y2_max - y2_min)
        hr_equiv = y1_min + power_norm * (y1_max - y1_min)
        return hr_equiv
    
    # Convert rated power line to Hr scale
    rated_power_in_hr_scale = power_to_hr_scale(rated_power)
    rated_power_hr_array = np.full_like(qr2_for_shading, rated_power_in_hr_scale, dtype=float)
    
    # Convert actual power values to Hr scale
    power_in_hr_scale = power_to_hr_scale(power_array)
    
    # Find the effective upper boundary
    effective_upper_boundary = np.minimum(surge_hr_array, rated_power_hr_array)
    
    # STEP 1: SHADE ENTIRE AREA AS NON-OPERATING ZONE (RED BASE LAYER)
    ax1.fill_between(
        qr2_for_shading, 
        y1_min, 
        y1_max,
        facecolor='red', 
        alpha=0.0, 
        zorder=1,
        interpolate=True
    )
    
    # STEP 2: SHADE OPERATING ZONE (GREEN) - OVERLAY ON TOP
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
    power_non_operating_zone_patch = Patch(facecolor='red', alpha=0.0)

    hr_legend_handles = [surge_line] + actual_hr_handles
    power_legend_handles = [rated_power_line] + power_handles

    all_handles = hr_legend_handles + power_legend_handles + [hr_operating_zone_patch, power_non_operating_zone_patch]
    all_labels = [h.get_label() for h in hr_legend_handles] + \
                 [h.get_label() for h in power_legend_handles] + \
                 [hr_operating_zone_patch.get_label(), power_non_operating_zone_patch.get_label()]

    ax1.legend(all_handles, all_labels,
               title='Curves & Zones (Hr - Left Axis, Power - Right Axis)',
               loc='lower center',
               bbox_to_anchor=(0.5, -0.3),
               ncol=3,
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
    st.set_page_config(
        page_title="Compressor Performance Analyzer",
        page_icon="🗜️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
        <style>
        .main {
            padding: 2rem;
        }
        .stButton>button {
            width: 100%;
            background-color: #0066CC;
            color: white;
            font-weight: bold;
            padding: 0.75rem;
            border-radius: 8px;
        }
        .stButton>button:hover {
            background-color: #0052A3;
        }
        </style>
        """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/compressor.png", width=100)
        st.title("📋 About")
        st.markdown("""
        ### Compressor Analyzer
        
        This tool analyzes compressor performance data and generates:
        
        **Features:**
        - Process curve generation
        - Performance map visualization
        - Multi-pressure analysis
        - Safe operating zone identification
        - Triple-axis plotting
        - Automated Excel reporting
        
        **Version:** 2.0  
        **Updated:** 2025
        """)
        
        st.markdown("---")
        st.markdown("""
        ### 📊 Output Plots
        
        **Plot 1: Process Curve**
        - Discharge Pressure vs Reduced Flow
        - Grouped by Suction Temperature
        - Dual X-axis (Qr² and Actual Flow)
        
        **Plot 2: Performance Map**
        - Reduced Head vs Reduced Flow
        - Power curves overlay
        - Surge line indication
        - Safe operating zone (green)
        - Triple-axis display
        """)
    
    # Main content
    st.title("🗜️ Compressor Performance Analysis and Reporting")
    st.markdown("### Automated Analysis Tool for Compressor Operating Envelopes")
    st.markdown("---")
    
    # Instructions in an expander
    with st.expander("📖 Instructions - Click to expand", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            #### 📥 Upload Requirements
            - File format: `.xlsx` (Excel)
            - Single or multiple pressure datasets
            - All required columns must be present
            
            #### 📊 Required Columns
            1. **Suction Pressure barg** - Compressor inlet pressure
            2. **Suction Temperature Deg C** - Inlet gas temperature
            3. **Discharge Pressure barg** - Compressor outlet pressure
            4. **Actual Gas Flow (AMCH)** or **(Am3/hr)** - Volumetric flow rate
            5. **Power (kW)** - Compressor power consumption
            6. **Actual Hr** - Actual reduced head
            7. **Qr2** - Reduced flow parameter (Qr²)
            8. **Surge HR** - Surge line reduced head values
            """)
        
        with col2:
            st.markdown("""
            #### 🔄 How to Use
            1. Enter the **Rated Power** (kW) for your compressor
            2. Upload your Excel file with performance data
            3. Click **Generate Analysis** button
            4. Review the generated plots for each pressure
            5. Download the complete Excel report
            
            #### 📈 Output Includes
            - **Raw Data Sheet**: Original uploaded data
            - **Process Curves**: One per unique suction pressure
            - **Performance Maps**: One per unique suction pressure
            - All plots embedded as images in Excel
            
            #### 🎯 Analysis Features
            - Automatic grouping by suction pressure
            - Temperature-based curve generation
            - Safe operating zone identification
            - Surge line visualization
            - Rated power limit indication
            """)
    
    st.markdown("---")
    
    # --- RATED POWER INPUT ---
    st.markdown("### ⚙️ Configuration")
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        rated_power = st.number_input(
            "Enter Compressor Rated Power (kW):",
            min_value=1,
            value=4481,
            step=10,
            help="This value defines the horizontal 'Rated Power' line on the Performance Map."
        )
        st.info(f"✓ Rated Power set to: **{rated_power} kW**")
    
    st.markdown("---")

    # --- Streamlit File Uploader ---
    st.markdown("### 📁 Upload Your Excel File")
    uploaded_file = st.file_uploader(
        "Choose an Excel file containing compressor performance data",
        type=['xlsx'],
        help="The file must contain all 8 required columns for analysis."
    )
    
    if uploaded_file is None:
        st.info("👆 Please upload an Excel file to begin analysis")
        
        # Sample data format
        with st.expander("💡 View Sample Data Format"):
            sample_df = pd.DataFrame({
                'Suction Pressure barg': [25.5, 25.5, 25.5, 25.5, 30.0, 30.0],
                'Suction Temperature Deg C': [35, 35, 40, 40, 35, 40],
                'Discharge Pressure barg': [85.2, 87.1, 86.5, 88.3, 90.1, 91.2],
                'Actual Gas Flow (Am3/hr)': [15000, 16000, 14500, 15800, 15500, 15200],
                'Power (kW)': [3800, 4100, 3750, 4050, 4200, 4150],
                'Actual Hr': [1250, 1280, 1240, 1275, 1300, 1290],
                'Qr2': [0.85, 0.92, 0.82, 0.90, 0.88, 0.86],
                'Surge HR': [1400, 1400, 1380, 1380, 1420, 1400]
            })
            st.dataframe(sample_df, use_container_width=True)
            st.caption("Note: Data should contain multiple rows for different operating conditions at various suction pressures and temperatures")
        
        # Expected output preview
        with st.expander("📊 Preview Expected Output Plots"):
            st.markdown("""
            #### Plot 1: Process Curve Example
            - **X-axis (bottom)**: Reduced Flow (Qr²)
            - **X-axis (top)**: Actual Gas Flow (Am³/hr) - orange labels
            - **Y-axis**: Discharge Pressure (barg)
            - **Legend**: Different colors for each suction temperature
            - **Title**: Includes suction pressure value
            
            #### Plot 2: Performance Map Example
            - **X-axis (bottom)**: Reduced Flow Rate (Qr²)
            - **X-axis (top)**: Actual Gas Flow (Am³/hr) - orange labels
            - **Y-axis (left, blue)**: Reduced Head (Hr)
            - **Y-axis (right, green)**: Power (kW)
            - **Green shaded area**: Safe operating zone
            - **Red line**: Surge line (left axis)
            - **Black dash-dot line**: Rated power limit (right axis)
            - **Colored solid lines**: Reduced head curves at different temperatures
            - **Colored dashed lines**: Power curves at different temperatures
            - **Legend**: All curves and zones labeled
            """)
            
            st.info("""
            💡 **Interpretation Tips:**
            - Operating points in the **green zone** are safe
            - Stay **below the surge line** (red) to avoid surge conditions
            - Stay **below rated power line** (black) to avoid overload
            - Each unique suction pressure gets its own set of plots
            """)
        
        return

    # Display file details
    st.success(f"✅ File uploaded: **{uploaded_file.name}**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("File Name", uploaded_file.name)
    with col2:
        st.metric("File Size", f"{uploaded_file.size / 1024:.2f} KB")
    with col3:
        st.metric("File Type", uploaded_file.type.split('.')[-1].upper())
    
    st.markdown("---")
    
    # --- Data Reading and Validation ---
    try:
        uploaded_file.seek(0)
        df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"❌ Error reading Excel file: {e}")
        with st.expander("💡 Troubleshooting Tips"):
            st.markdown("""
            1. Ensure the file is a valid Excel file (.xlsx format)
            2. Check that the file is not corrupted
            3. Try opening and re-saving the file in Excel
            4. Verify that the file is not password-protected
            """)
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
        st.error(f"❌ Missing required columns: **{', '.join(missing_columns)}**")
        
        with st.expander("📋 Show Available Columns"):
            st.write("Columns found in your file:")
            st.code('\n'.join([f"- {col}" for col in df.columns]), language="text")
        
        with st.expander("✅ Show Required Columns"):
            st.write("Columns needed for analysis:")
            st.code('\n'.join([f"- {col}" for col in required_columns]), language="text")
        
        st.warning("Please update your Excel file to include all required columns and upload again.")
        return

    # Data preview
    with st.expander("👀 Preview Uploaded Data (first 10 rows)"):
        st.dataframe(df.head(10), use_container_width=True)
        st.info(f"Total rows: {len(df)}")

    # Analysis
    unique_pressures = sorted(df['Suction Pressure barg'].unique())
    unique_temps = sorted(df['Suction Temperature Deg C'].unique())
    
    st.markdown("### 📊 Data Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Data Points", len(df))
    with col2:
        st.metric("Unique Pressures", len(unique_pressures))
    with col3:
        st.metric("Unique Temperatures", len(unique_temps))
    with col4:
        st.metric("Total Plots", len(unique_pressures) * 2)
    
    st.info(f"📍 **Suction Pressures found:** {', '.join([f'{p} barg' for p in unique_pressures])}")
    st.info(f"🌡️ **Suction Temperatures found:** {', '.join([f'{t}°C' for t in unique_temps])}")
    
    st.markdown("---")
    
    # Generate button
    st.markdown("### 🚀 Generate Analysis")
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        generate_button = st.button("🔄 Generate Analysis & Plots", type="primary", use_container_width=True)
    
    if not generate_button:
        st.info("👆 Click the button above to generate plots and create the Excel report")
        return
    
    # --- Generate Plots and Excel ---
    all_plot_data = []
    excel_buffer = BytesIO()
    
    try:
        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
            
            # 1. Write the Raw Data
            df.to_excel(writer, sheet_name='Raw Data Input', index=False)
            st.success("✓ Raw data written to Excel")
            
            workbook = writer.book

            # Loop Through Each Suction Pressure and Generate Plots
            st.markdown("### 📈 Generating Plots...")
            plot_progress = st.progress(0, text="Initializing...")
            
            for i, pressure in enumerate(unique_pressures):
                with st.spinner(f"Processing suction pressure: **{pressure} barg**..."):
                    df_pressure = df[df['Suction Pressure barg'] == pressure].copy()
                    df_sorted_pressure = df_pressure.sort_values(by='Qr2').reset_index(drop=True)
                    
                    # Plot Set 1: Process Curve
                    process_name, process_buffer = plot_qr2_vs_discharge_pressure_by_temp(
                        df_pressure, df_sorted_pressure, pressure
                    )
                    all_plot_data.append((process_name, process_buffer))
                    
                    # Plot Set 2: Performance Map 
                    performance_name, performance_buffer = plot_superimposed_map_triple_axis(
                        df_pressure, df_sorted_pressure, rated_power, pressure
                    )
                    all_plot_data.append((performance_name, performance_buffer))
                    
                    # Display plots in Streamlit
                    st.success(f"✓ Generated plots for pressure: **{pressure} barg**")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"#### Process Curve - {pressure} barg")
                        process_buffer.seek(0)
                        st.image(process_buffer, use_container_width=True)
                    
                    with col2:
                        st.markdown(f"#### Performance Map - {pressure} barg")
                        performance_buffer.seek(0)
                        st.image(performance_buffer, use_container_width=True)
                    
                    st.markdown("---")
                    
                    # Update progress
                    plot_progress.progress(
                        (i + 1) / len(unique_pressures), 
                        text=f"Generated plots for pressure {i+1}/{len(unique_pressures)}: {pressure} barg"
                    )

            plot_progress.empty()
            st.success("🎉 All plots generated successfully!")
            
            # 2. Embed all plots in Excel
            st.info("📝 Embedding plots into Excel file...")
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
            st.markdown("---")
            st.success("### ✅ Analysis Complete!")
            
            output_filename = f'Compressor_Performance_Output_{os.path.splitext(uploaded_file.name)[0]}.xlsx'
            
            st.markdown("""
            **Your Excel report includes:**
            - ✓ Raw data input sheet
            - ✓ Process curve for each suction pressure
            - ✓ Performance map for each suction pressure
            - ✓ All plots embedded as high-resolution images
            """)
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.download_button(
                    label="📥 Download Complete Analysis Excel File",
                    data=excel_buffer,
                    file_name=output_filename,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            
    except Exception as e:
        st.error(f"❌ An error occurred during processing: {str(e)}")
        with st.expander("🔍 Error Details"):
            st.code(str(e), language="text")
        with st.expander("💡 Troubleshooting"):
            st.markdown("""
            1. Verify all data values are numeric (no text in numeric columns)
            2. Check for missing values (NaN) in critical columns
            3. Ensure there are multiple data points per pressure
            4. Verify Qr2 values are properly calculated
            5. Check that temperature values are consistent
            """)

# Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🗜️ Compressor Performance Analyzer v2.0 | Built with Streamlit</p>
        <p style='font-size: 0.8em;'>For support or questions, contact your system administrator</p>
    </div>
    """, unsafe_allow_html=True)
