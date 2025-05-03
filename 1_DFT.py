import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
from itertools import product, combinations
from pyDOE2 import fracfact
import seaborn as sns

# Custom CSS for styling
def set_custom_style():
    st.markdown("""
    <style>
        /* Lighten sidebar background */
        [data-testid="stSidebar"] {
            background-color: #f8f9fa !important;
            border-right: 1px solid #e0e0e0;
        }
        
        /* Dark text for light sidebar */
        [data-testid="stSidebar"] .stRadio > label,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] .stMarkdown {
            color: #333333 !important;
        }
        
        /* Sidebar hover effects */
        [data-testid="stSidebar"] .stRadio > div:hover {
            background-color: #e9ecef;
            border-radius: 5px;
        }
    </style>
    """, unsafe_allow_html=True)

### Main Functions

st.set_page_config(page_title="DOE & ANOVA App", layout="wide")

st.sidebar.title("Design of Experiments")
st.sidebar.markdown("----")

selected_page = st.sidebar.radio(
    "Navigation Menu",
    ["6 Factors", "3 Factors"]
)
set_custom_style()

if selected_page == "6 Factors":
   

    st.title("Resolution V Factorial Design Application (6 Factors)")
    st.markdown("This application generates a 6-factor design of experiments (DOE) and performs ANOVA analysis. The goal is to identify the \
                most significant factors affecting the energy of a system.")
    # Sidebar inputs
    st.subheader("Step 1: Adjust Factor Ranges and Generate Design of Experiments (DOE)")
        # Define 2 columns for better layout
    with st.expander("🔍 Filter Settings", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            ecut_range = st.slider("ecut (Ry)", 20, 100, (20, 80))
            degauss_range = st.slider("degauss (Ry)", 0.001, 0.3, (0.01, 0.20))
            grid_range = st.slider("grid", 2, 30, (2, 24))

        with col2:
            celldm3_range = st.slider("Celldm3", 0.0, 20.0, (5.0, 15.0))
            celldm1_range = st.slider("Celldm1", 2.0, 12.0, (3.0, 8.0))
            smearing_options = st.multiselect(
                "Smearing options", ['m-v', 'gaussian'], default=['m-v', 'gaussian']
            )

    # Factor names and order
    factors = ['ecut', 'smearing', 'degauss', 'grid', 'Celldm3', 'Celldm1']

    # Step 1: Generate Resolution V design (f = a*b*c)
    design = fracfact('a b c d e abc')  # 2^(6-1) = 32 runs

    # Step 2: Create coded DataFrame
    df_coded = pd.DataFrame(design, columns=factors)

    # Step 3: Map numeric coded values (-1, +1) to actual ranges
    def scale_values(x, low, high):
        return ((x + 1) / 2) * (high - low) + low

    df = df_coded.copy()
    df.index += 1
    df['ecut'] = scale_values(df['ecut'], *ecut_range)
    df['degauss'] = scale_values(df['degauss'], *degauss_range)
    df['grid'] = scale_values(df['grid'], *grid_range)
    df['Celldm3'] = scale_values(df['Celldm3'], *celldm3_range)
    df['Celldm1'] = scale_values(df['Celldm1'], *celldm1_range)

    # Step 4: Map smearing coded value (-1, +1) to categories
    df['smearing'] = df['smearing'].map({-1: smearing_options[0], 1: smearing_options[1]})

    # Step 5: Add empty energy column
    df["energy"] = ""  # For manual or simulation-based input

    st.subheader("Design of Experiments Table")
    st.dataframe(df, use_container_width=True)

    # Download link
    st.subheader("Step 2: Download the following File and run in Quantum EXPRESSO")
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download DOE", data=csv, file_name="doe.csv", mime="text/csv")

    # Step 3: Upload completed DOE file
    st.subheader("Step 3: Upload DOE with Energy Values")
    uploaded_file = st.file_uploader("Upload CSV with energy values added:", type="csv")
    if uploaded_file is not None:
        df_uploaded = pd.read_csv(uploaded_file)
        st.dataframe(df_uploaded)
    else:
        st.info("ℹ️ No file uploaded. Using default backup file.")
        try:
            df_uploaded = pd.read_csv(r"Design of Experiments - Sheet2_ff6_v1.csv")
            st.success("✅ Loaded backup_doe.csv as fallback.")
        except FileNotFoundError:
            st.error("❌ No file uploaded and 'backup_doe.csv' not found.")
            df_uploaded = None
    if df_uploaded is not None:
        st.dataframe(df_uploaded)

        if st.button("Run ANOVA"):
            # Ensure categorical format
            df_uploaded['smearing'] = df_uploaded['smearing'].astype(str)
            for factor in factors:
                df_uploaded[factor] = df_uploaded[factor].astype(str)

            # Build model formula
            factors_all = factors + ['smearing']
            main_effects = ' + '.join(factors_all)
            two_way = ' + '.join([f'{a}:{b}' for a, b in combinations(factors_all, 2)])
            formula = f'energy ~ {main_effects} + {two_way}'

            model = ols(formula, data=df_uploaded).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)

            st.subheader("ANOVA Table")
            st.dataframe(anova_table)

            # Optional: Display full model summary stats
            st.subheader("📈 Model Summary Statistics")
            st.markdown(f"""
            - **R-squared**: {model.rsquared:.4f}  
            - **Adjusted R-squared**: {model.rsquared_adj:.4f}  
            - **F-statistic**: {model.fvalue:.4f}  
            - **Prob (F-statistic)**: {model.f_pvalue:.4e}  
            - **Residual Std Error (RMSE)**: {np.sqrt(model.mse_resid):.4f}  
            - **Degrees of Freedom**: {int(model.df_resid)}  
            """)

            # Pareto plot
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

            # Significant effects
            st.subheader("Significant Effects (p < 0.05)")
            significant = anova_table[anova_table['PR(>F)'] < 0.05]
            st.write(significant)

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
        st.subheader(f"📋 Full Factorial Design ({len(df)} Runs)")
        st.dataframe(df)

        # Download
        st.subheader("Step 2: Download the following File and run in Quantum ESPRESSO")
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download DOE", data=csv, file_name="doe.csv", mime="text/csv")

        # Upload section
        st.subheader("Step 3: Upload Energy Results for Analysis")
        uploaded_file = st.file_uploader("Upload DOE results (CSV with 'energy' column filled)", type=["csv"])

        if uploaded_file is not None:
            df_uploaded = pd.read_csv(uploaded_file)
            df_uploaded.columns = df_uploaded.columns.str.strip()  # Clean up column names
            #st.success("✅ File successfully loaded.")
        else:
            st.info("ℹ️ No file uploaded. Using default backup file.") 
            try:
                df_uploaded = pd.read_csv(r"Design of Experiments - Sheet2_ff3_v1.csv")
                st.success("✅ Loaded backup_doe.csv as fallback.")
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
                st.dataframe(df_uploaded)

                # Convert categorical column(s)
                for col in selected_factors:
                    if col == "smearing":
                        df_uploaded[col] = df_uploaded[col].astype(str)

                # Run ANOVA
                st.subheader("📊 ANOVA Results")
                try:
                    formula = f"energy ~ {' * '.join(selected_factors)}"
                    model = ols(formula, data=df_uploaded).fit()
                    anova_table = sm.stats.anova_lm(model, typ=2)
                    st.write(anova_table)

                    st.markdown(f"**R²:** {model.rsquared:.4f}")
                    st.markdown(f"**Adjusted R²:** {model.rsquared_adj:.4f}")
                    st.markdown(f"**Model p-value:** {model.f_pvalue:.4e}")

                    # Interactive Lineplot
                    st.subheader("📉 Explore Energy Trends by Factor")
                    plot_x = st.selectbox("Select X-axis variable:", 'Celldm1')
                    plot_hue = st.selectbox("Select grouping (hue):", [f for f in selected_factors if f != plot_x])

                    fig2, ax2 = plt.subplots()
                    sns.lineplot(data=df_uploaded, x=plot_x, y="energy", hue=plot_hue, marker="o", ax=ax2)
                    ax2.set_title(f"Energy vs {plot_x} grouped by {plot_hue}")
                    ax2.set_ylabel("Total Energy")
                    st.pyplot(fig2)

                except Exception as e:
                    st.error(f"❌ ANOVA failed: {str(e)}")

