import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
from itertools import product, combinations
import seaborn as sns
from PIL import Image
from matplotlib import cm 
from scipy.optimize import minimize
import plotly.graph_objects as go
import itertools

# Custom CSS for styling
def set_custom_style():
    st.markdown("""
    <style>
        [data-testid="stSidebar"] {
            background-color: #f8f9fa !important;
            border-right: 1px solid #e0e0e0;
        }
        [data-testid="stSidebar"] .stRadio > label,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] .stMarkdown {
            color: #333333 !important;
        }
        [data-testid="stSidebar"] .stRadio > div:hover {
            background-color: #e9ecef;
            border-radius: 5px;
        }
    </style>
    """, unsafe_allow_html=True)

# Replacement for pyDOE2.fracfact
def generate_fracfact_design():
    base_design = list(product([-1, 1], repeat=5))
    df = pd.DataFrame(base_design, columns=['a', 'b', 'c', 'd', 'e'])
    df['f'] = df['a'] * df['b'] * df['c']
    df.columns = ['ecut', 'smearing', 'degauss', 'grid', 'Celldm3', 'Celldm1']
    return df

st.set_page_config(page_title="DOE & ANOVA App", layout="wide")

st.sidebar.title("Design of Experiments")
st.sidebar.markdown("----")

selected_page = st.sidebar.radio("Navigation Menu", ["6 Factors", "3 Factors", "2 Factors"])
set_custom_style()

if selected_page == "6 Factors":
    st.title("Factorial Design Application (Res V)")
    st.markdown("This application generates a 6-factor design of experiments (DOE) and performs ANOVA analysis.")

    st.subheader("Step 1: Adjust Factor Ranges and Generate Design of Experiments (DOE)")
    with st.expander("\U0001F50D Filter Settings", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            ecut_range = st.slider("ecut (Ry)", 20, 100, (20, 80))
            degauss_range = st.slider("degauss (Ry)", 0.001, 0.3, (0.01, 0.20))
            grid_range = st.slider("grid", 2, 30, (2, 24))
        with col2:
            celldm3_range = st.slider("Celldm3", 0.0, 20.0, (5.0, 15.0))
            celldm1_range = st.slider("Celldm1", 2.0, 12.0, (3.0, 8.0))
            smearing_options = st.multiselect("Smearing options", ['m-v', 'gaussian'], default=['m-v', 'gaussian'])

    factors = ['ecut', 'smearing', 'degauss', 'grid', 'Celldm3', 'Celldm1']
    df_coded = generate_fracfact_design()

    def scale_values(x, low, high):
        return ((x + 1) / 2) * (high - low) + low

    df = df_coded.copy()
    df.index += 1
    df['ecut'] = scale_values(df['ecut'], *ecut_range)
    df['degauss'] = scale_values(df['degauss'], *degauss_range)
    df['grid'] = scale_values(df['grid'], *grid_range)
    df['Celldm3'] = scale_values(df['Celldm3'], *celldm3_range)
    df['Celldm1'] = scale_values(df['Celldm1'], *celldm1_range)
    df['smearing'] = df['smearing'].map({-1: smearing_options[0], 1: smearing_options[1]})
    df["energy"] = ""

    with st.expander("🧪 DOE Table"):
        st.subheader("Design of Experiments Table")
        st.dataframe(df, use_container_width=True)

    st.subheader("Step 2: Download the following File")
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("🚀 Download DOE", data=csv, file_name="doe.csv", mime="text/csv")

    st.subheader("Step 3: Make a bashfile and encode the DOE in Quantum EXPRESSO")
    with st.expander("💻 Sample Bashfile Code"):
        image = Image.open("fracfac_code.png")
        st.image(image, caption="DOE bashfile example", use_container_width=True)

    st.subheader("Step 4: Upload DOE with Energy Values")
    uploaded_file = st.file_uploader("Upload CSV with energy values added:", type="csv")
    if uploaded_file is not None:
        df_uploaded = pd.read_csv(uploaded_file)
        with st.expander("🧮 Data Table"):
            st.dataframe(df_uploaded)
    else:
        st.info("\u2139\ufe0f No file uploaded. Using default backup file.")
        try:
            df_uploaded = pd.read_csv(r"Design of Experiments - Sheet2_ff6_v1.csv")
            #st.success("\u2705 Loaded backup_doe.csv as fallback.")
        except FileNotFoundError:
            st.error("\u274c No file uploaded and 'backup_doe.csv' not found.")
            df_uploaded = None

    if df_uploaded is not None:
        with st.expander("🧮 Data Table"):
            st.dataframe(df_uploaded)

        if st.button("Run ANOVA"):
            df_uploaded['smearing'] = df_uploaded['smearing'].astype(str)
            for factor in factors:
                df_uploaded[factor] = df_uploaded[factor].astype(str)

            main_effects = ' + '.join(factors)
            two_way = ' + '.join([f'{a}:{b}' for a, b in combinations(factors, 2)])
            formula = f'energy ~ {main_effects} + {two_way}'

            model = ols(formula, data=df_uploaded).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)

            with st.expander("📊 ANOVA Results"):
                st.subheader("ANOVA Table")
                st.dataframe(anova_table)

            st.subheader("\U0001F4C8 Model Summary Statistics")
            st.markdown(f"""
            - **R-squared**: {model.rsquared:.4f}  
            - **Adjusted R-squared**: {model.rsquared_adj:.4f}  
            - **F-statistic**: {model.fvalue:.4f}  
            - **Prob (F-statistic)**: {model.f_pvalue:.4e}  
            - **Residual Std Error (RMSE)**: {np.sqrt(model.mse_resid):.4f}  
            - **Degrees of Freedom**: {int(model.df_resid)}  
            """)

            anova_table['-log10(p)'] = -np.log10(anova_table['PR(>F)'].replace(0, 1e-10))
            sorted_anova = anova_table.sort_values('-log10(p)', ascending=False)

            fig, ax = plt.subplots(figsize=(14, 6))
            sorted_anova['-log10(p)'].plot(kind='bar', ax=ax)
            ax.axhline(y=1.3, color='red', linestyle='--', label='p = 0.05')
            ax.set_ylabel('-log10(p-value)')
            ax.set_title('Pareto Chart of Effects')
            plt.xticks(rotation=90)
            ax.legend()
            st.pyplot(fig)

            st.subheader("Significant Effects (p < 0.05)")
            significant = anova_table[anova_table['PR(>F)'] < 0.05]
            st.write(significant)
# Note: The 3-factor section remains unchanged. Update similarly if needed.


elif selected_page == "3 Factors":
    st.title("Full Factorial Application (3³ Design)")
    st.markdown(
        "This application generates a 3-factor design of experiments (DOE) and performs ANOVA analysis. "
        "The goal is to identify the relationship between factors and to test the presence of curvature."
    )

    st.subheader("Step 1: Adjust Factor Ranges and Generate Design of Experiments (DOE)")

    with st.expander("🔍 Filter Settings", expanded=True):
        available_factors = ['ecut', 'degauss', 'grid', 'Celldm3', 'Celldm1', 'smearing']
        selected_factors = st.multiselect(
            "Select 3 Factors for the DOE:",
            available_factors,
            default=['degauss', 'Celldm1','smearing']
        )

        if len(selected_factors) != 3:
            st.warning("⚠️ Please select exactly 3 factors.")
        else:
            col1, col2 = st.columns(2)
            factor_inputs = {}

            with col1:
                if 'ecut' in selected_factors:
                    factor_inputs['ecut'] = st.slider("ecut (Ry)", 20, 100, (20, 80))
                if 'degauss' in selected_factors:
                    factor_inputs['degauss'] = st.slider("degauss (Ry)", 0.001, 0.3, (0.01, 0.20))
                if 'grid' in selected_factors:
                    factor_inputs['grid'] = st.slider("grid", 2, 30, (2, 24))

            with col2:
                if 'Celldm3' in selected_factors:
                    factor_inputs['Celldm3'] = st.slider("Celldm3", 0.0, 20.0, (5.0, 15.0))
                if 'Celldm1' in selected_factors:
                    factor_inputs['Celldm1'] = st.slider("Celldm1", 2.0, 12.0, (3.0, 8.0))
                if 'smearing' in selected_factors:
                    factor_inputs['smearing'] = st.multiselect(
                        "Smearing options", ['m-v', 'gaussian'], default=['m-v', 'gaussian']
                    )

    # Generate levels per factor
    def get_levels(name, value):
        if name == 'smearing':
            return value  # return list of categories
        else:
            low, high = value
            return np.linspace(low, high, 3).round(4)

    # Build combinations and DataFrame
    if len(selected_factors) == 3:
        levels_dict = {factor: get_levels(factor, factor_inputs[factor]) for factor in selected_factors}
        factor_combinations = list(product(*levels_dict.values()))
        df = pd.DataFrame(factor_combinations, columns=selected_factors)
        df.index += 1
        df['energy'] = ""  # Placeholder
        with st.expander("🧪 Design of Experiments"):
            st.subheader(f"📋 Full Factorial Design ({len(df)} Runs)")
            st.dataframe(df)

        # Download
        st.subheader("Step 2: Download the following File")
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("🚀 Download DOE", data=csv, file_name="doe.csv", mime="text/csv")

        st.subheader("Step 3: Make a bashfile and encode the DOE in Quantum EXPRESSO")
        with st.expander("💻 Sample Bashfile Code"):
            image = Image.open("fullfrac_code.png")
            st.image(image, caption="DOE bashfile example", use_container_width=True)

        # Upload section
        st.subheader("Step 4: Upload Energy Results for Analysis")
        uploaded_file = st.file_uploader("Upload DOE results (CSV with 'energy' column filled)", type=["csv"])

        if uploaded_file is not None:
            df_uploaded = pd.read_csv(uploaded_file)
            df_uploaded.columns = df_uploaded.columns.str.strip()  # Clean up column names
            #st.success("✅ File successfully loaded.")
        else:
            st.info("ℹ️ No file uploaded. Using default backup file.") 
            try:
                df_uploaded = pd.read_csv(r"Design of Experiments - Sheet2_ff3_v1.csv")
                #st.success("✅ Loaded backup_doe.csv as fallback.")
            except FileNotFoundError:
                st.error("❌ No file uploaded and 'backup_doe.csv' not found.")
                df_uploaded = None 
        #if df_uploaded is not None:
        #    st.dataframe(df_uploaded)

            if 'energy' not in df_uploaded.columns:
                st.error("❌ Uploaded file must contain an 'energy' column.")
            else:
                # Validate selected factors exist in uploaded data
                missing_cols = [f for f in selected_factors if f not in df_uploaded.columns]
                if missing_cols:
                    st.error(f"❌ The following selected factors are missing from the uploaded file: {missing_cols}")
                    st.stop()

                #st.success("✅ File successfully loaded.")
                with st.expander("🧮 Experimental Results"):
                    st.dataframe(df_uploaded)

                # Convert categorical column(s)
                for col in selected_factors:
                    if col == "smearing":
                        df_uploaded[col] = df_uploaded[col].astype(str)

                # Run ANOVA
                with st.expander("📊 ANOVA Results"):
                    st.subheader("📊 ANOVA Results")
                    try:
                        formula = f"energy ~ {' * '.join(selected_factors)}"
                        model = ols(formula, data=df_uploaded).fit()
                        anova_table = sm.stats.anova_lm(model, typ=2)
                        st.write(anova_table.round(4))

                        st.markdown(f"**R²:** {model.rsquared:.4f}")
                        st.markdown(f"**Adjusted R²:** {model.rsquared_adj:.4f}")
                        st.markdown(f"**Model p-value:** {model.f_pvalue:.4e}")

                        anova_table['-log10(p)'] = -np.log10(anova_table['PR(>F)'].replace(0, 1e-10))
                        sorted_anova = anova_table.sort_values('-log10(p)', ascending=False)

                        fig, ax = plt.subplots(figsize=(14, 6))
                        sorted_anova['-log10(p)'].plot(kind='bar', ax=ax)
                        ax.axhline(y=1.3, color='red', linestyle='--', label='p = 0.05')
                        ax.set_ylabel('-log10(p-value)')
                        ax.set_title('Pareto Chart of Effects')
                        plt.xticks(rotation=90)
                        ax.legend()
                        st.pyplot(fig)

                        # Interactive Lineplot
                        st.subheader("📉 Explore Energy Trends by Factor")
                        col1, col2 = st.columns(2)
                        factor_inputs = {}

                        with col1:
                            plot_x = st.selectbox("Select X-axis variable:", 'Celldm1')
                        with col2:
                            plot_hue = st.selectbox("Select grouping (hue):", [f for f in selected_factors if f != plot_x])

                        fig2, ax2 = plt.subplots(figsize=(8, 5))
                        sns.lineplot(data=df_uploaded, x=plot_x, y="energy", hue=plot_hue, marker="o", ax=ax2)
                        ax2.set_title(f"Energy vs {plot_x} grouped by {plot_hue}")
                        ax2.set_ylabel("Total Energy")
                        st.pyplot(fig2)

                    except Exception as e:
                        st.error(f"❌ ANOVA failed: {str(e)}")

elif selected_page == "2 Factors":
    st.title("Response Surface Methodology (CCD)")
    st.markdown(
        "This application generates a 2-factor RSM design and performs regression and ANOVA analysis. "
        "The goal is to model curvature and interactions using a Central Composite Design (CCD)."
    )

    st.subheader("Step 1: Choose 2 Factors and Set Ranges")

    with st.expander("⚙️ RSM Factor Selection", expanded=True):
        available_factors = ['ecut', 'degauss', 'grid', 'Celldm3', 'Celldm1']
        selected_factors = st.multiselect(
            "Select exactly 2 numeric factors for RSM:",
            available_factors,
            default=['Celldm1', 'Celldm3']
        )

        if len(selected_factors) != 2:
            st.warning("⚠️ Please select exactly 2 numeric factors.")
        else:
            col1, col2 = st.columns(2)
            factor_ranges = {}
            with col1:
                if 'Celldm1' in selected_factors:
                    factor_ranges['Celldm1'] = st.slider("Celldm1", 3.0, 12.0, (4.0, 5.5))
                if 'ecut' in selected_factors:
                    factor_ranges['ecut'] = st.slider("ecut (Ry)", 20, 100, (20, 80))
                if 'grid' in selected_factors:
                    factor_ranges['grid'] = st.slider("grid", 2, 30, (2, 24))

            with col2:
                if 'Celldm3' in selected_factors:
                    factor_ranges['Celldm3'] = st.slider("Celldm3", 2.0, 15.0, (9.0, 11.0))
                if 'degauss' in selected_factors:
                    factor_ranges['degauss'] = st.slider("degauss (Ry)", 0.001, 0.3, (0.01, 0.20))
                if 'smearing' in selected_factors:
                    factor_ranges['smearing'] = st.multiselect(
                        "Smearing options", ['m-v', 'gaussian'], default=['m-v', 'gaussian']
                    )

            # Central Composite Design
    num_center_points = 3

    k = len(selected_factors)
    alpha = np.sqrt(k)  # Rotatability, same as in pyDOE2
    levels = [-1, 1]

    # 1. Full factorial design points
    factorial_points = list(itertools.product(levels, repeat=k))

    # 2. Axial points (±α along each axis)
    axial_points = []
    for i in range(k):
        pos = [0] * k
        neg = [0] * k
        pos[i] = alpha
        neg[i] = -alpha
        axial_points.append(tuple(pos))
        axial_points.append(tuple(neg))

    # 3. Center points
    center_points = [(0,) * k] * num_center_points

    # Combine all CCD points
    ccd = factorial_points + axial_points + center_points
    design_df = pd.DataFrame(ccd, columns=selected_factors)

    # Rescale from coded [-1, 1] (or ±α) to real-world values
    for i, factor in enumerate(selected_factors):
        low, high = factor_ranges[factor]
        design_df[factor] = ((design_df[factor] + 1) / 2) * (high - low) + low
        design_df[factor] = design_df[factor].round(1)

    # Add placeholder column for results
    design_df['energy'] = ""
    design_df.index += 1  # Start index at 1

    with st.expander("📋 RSM Design Matrix"):
        st.write(f"📈 Total Runs: {len(design_df)}")
        st.dataframe(design_df)

    # Download CSV
    st.subheader("Step 2: Download the RSM Design")
    csv = design_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download RSM CSV", data=csv, file_name="rsm_design.csv", mime="text/csv")

    # File Upload
    st.subheader("Step 3: Upload Results for Analysis")
    uploaded_file = st.file_uploader("Upload completed CSV with 'energy' column", type=["csv"])

    if uploaded_file:
        df_uploaded = pd.read_csv(uploaded_file)
        if "energy" not in df_uploaded.columns:
            st.error("❌ Uploaded file must contain an 'energy' column.")
        else:
            st.success("✅ File successfully loaded.")
            st.dataframe(df_uploaded)
    
            # Regression and ANOVA
    with st.expander("📊 RSM ANOVA & Model Fit"):
        try:
            # --- 1. Fit the Model ---
            formula = f"energy ~ {selected_factors[0]} + {selected_factors[1]} + " \
                    f"I({selected_factors[0]}**2) + I({selected_factors[1]}**2) + " \
                    f"{selected_factors[0]}:{selected_factors[1]}"
            model = ols(formula, data=df_uploaded).fit()
            
            # --- 2. Display ANOVA and Summary Stats ---
            anova_table = sm.stats.anova_lm(model, typ=2)
            st.write(anova_table.round(4))
            st.markdown(f"**R²:** {model.rsquared:.4f}")
            st.markdown(f"**Adjusted R²:** {model.rsquared_adj:.4f}")
            st.markdown(f"**Model p-value:** {model.f_pvalue:.4e}")

            # --- 3. Generate Meshgrid and Predictions ---
            x = df_uploaded[selected_factors[0]]
            y = df_uploaded[selected_factors[1]]
            xx, yy = np.meshgrid(np.linspace(x.min(), x.max(), 100),
                                np.linspace(y.min(), y.max(), 100))
            
            data_grid = pd.DataFrame({
                selected_factors[0]: xx.ravel(),
                selected_factors[1]: yy.ravel(),
                f'I({selected_factors[0]}**2)': xx.ravel() ** 2,
                f'I({selected_factors[1]}**2)': yy.ravel() ** 2,
                f'{selected_factors[0]}:{selected_factors[1]}': xx.ravel() * yy.ravel()
            })
            
            zz = model.predict(data_grid).values.reshape(xx.shape)

            # --- 4. Optimize Response Surface ---
            def response_surface(x_vals):
                df_pred = pd.DataFrame({
                    selected_factors[0]: [x_vals[0]],
                    selected_factors[1]: [x_vals[1]],
                    f'I({selected_factors[0]}**2)': [x_vals[0]**2],
                    f'I({selected_factors[1]}**2)': [x_vals[1]**2],
                    f'{selected_factors[0]}:{selected_factors[1]}': [x_vals[0]*x_vals[1]]
                })
                return model.predict(df_pred)[0]

            initial_guess = [x.mean(), y.mean()]
            bounds = [(x.min(), x.max()), (y.min(), y.max())]
            result = minimize(response_surface, initial_guess, method='L-BFGS-B', bounds=bounds)

            optimal_factors = result.x
            predicted_energy_at_optimal = result.fun

            # --- 5. Display Optimization Results ---
            st.markdown("### Optimum Factor Levels and Predicted Energy")
            st.write(f"**Optimum {selected_factors[0]}:** {optimal_factors[0]:.4f}")
            st.write(f"**Optimum {selected_factors[1]}:** {optimal_factors[1]:.4f}")
            st.write(f"**Predicted Energy at Optimum:** {predicted_energy_at_optimal:.4f}")

            # --- 6. Matplotlib Contour Plot ---
            fig, ax = plt.subplots(figsize=(6, 5))
            contour = ax.contourf(xx, yy, zz, levels=30, cmap='viridis')
            fig.colorbar(contour, ax=ax)
            ax.set_xlabel(selected_factors[0])
            ax.set_ylabel(selected_factors[1])
            ax.set_title("Contour Plot")
            ax.plot(optimal_factors[0], optimal_factors[1], 'ro', label='Optimum')
            ax.legend()

            # --- 7. Plotly 3D Surface Plot ---
            fig3d = go.Figure(data=[go.Surface(z=zz, x=xx, y=yy, colorscale='Viridis')])
            fig3d.add_trace(go.Scatter3d(
                x=[optimal_factors[0]],
                y=[optimal_factors[1]],
                z=[predicted_energy_at_optimal],
                mode='markers+text',
                marker=dict(color='red', size=6),
                text=["Optimum"],
                textposition="top center"
            ))
            fig3d.update_layout(
                title="3D Response Surface",
                scene=dict(
                    xaxis_title=selected_factors[0],
                    yaxis_title=selected_factors[1],
                    zaxis_title='Predicted Energy'
                )
            )

            # --- 8. Display Plots ---
            col1, col2 = st.columns(2)
            with col1:
                st.pyplot(fig)
            with col2:
                st.plotly_chart(fig3d)

        except Exception as e:
            st.error(f"Model fitting or plotting failed: {e}")