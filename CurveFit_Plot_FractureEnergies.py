import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.lines as mlines

rcParams['font.family'] = 'Arial'

# Set style parameters
rcParams['axes.linewidth'] = 1.2
rcParams['lines.linewidth'] = 1.2
rcParams['grid.linewidth'] = 1.2
rcParams['xtick.major.width'] = 1.2
rcParams['ytick.major.width'] = 1.2

# New grid settings
rcParams['axes.grid'] = True
rcParams['grid.alpha'] = 0.5
rcParams['grid.color'] = 'gray'
rcParams['grid.linestyle'] = '--'
rcParams['grid.linewidth'] = 0.5

# Systems with BCP additive
Et_BCP = [(2850, 57), (2660, 28), (2590, 41), (2140, 41), (2000, 30), (1650, 41)]
σm_BCP = [(64, 0.5), (59, 0.1), (56, 0.7), (34, 0.9), (27, 0.5), (17, 0.8)]
εm_BCP = [(4.3, 0.01), (4.2, 0.04), (3.9, 0.05), (2.8, 0.06), (2.5, 0.22), (1.8, 0.32)]
KIc_BCP = [(1.81, 0.2), (2.04, 0.1), (1.97, 0.1), (1.87, 0.2), (1.63, 0.1), (0.93, 0.2)]
GIc_BCP = [(1.01, 0.03), (1.37, 0.06), (1.32, 0.03), (1.45, 0.11), (1.17, 0.28), (0.48, 0.18)]

# Systems with CSR additive
Et_CSR = [(2850, 17), (2800, 11), (2770, 34), (2690, 40), (2650, 30), (2610, 34)]
σm_CSR = [(65, 0.3), (63, 0.5), (61, 0.3), (59, 0.3), (58, 0.2), (55, 0.2)]
εm_CSR = [(4.3, 0.08), (4.3, 0.12), (4.3, 0.03), (4.2, 0.03), (4.2, 0.05), (4.2, 0.01)]
KIc_CSR = [(1.19, 0.1), (1.25, 0.1), (1.54, 0.1), (1.65, 0.1), (1.72, 0.1), (1.77, 0.1)]
GIc_CSR = [(0.43, 0.11), (0.49, 0.02), (0.75, 0.06), (0.89, 0.10), (0.95, 0.04), (1.05, 0.02)]

# Systems with both BCP and CSR additives
Et_BCP_CSR = [(2850, 33), (2800, 11), (2700, 50)]
σm_BCP_CSR = [(67, 0.3), (62, 0.9), (60, 0.9)]
εm_BCP_CSR = [(4.4, 0.01), (4.3, 0.03), (4.2, 0.03)]
KIc_BCP_CSR = [(1.73, 0.1), (1.94, 0.1), (1.99, 0.1)]
GIc_BCP_CSR = [(0.92, 0.11), (1.18, 0.04), (1.28, 0.13)]

# Function to extract actual values and std dev from the data lists
def extract_values_std(data_list):
    values, std_devs = zip(*data_list)
    return list(values), list(std_devs)

# Percentage weight of the additive for BCP and CSR (since they have the same percentages)
percentages = [2, 4, 6, 8, 10, 12]

# Extract values and standard deviations for each property
Et_values_BCP, Et_std_BCP = extract_values_std(Et_BCP)
σm_values_BCP, σm_std_BCP = extract_values_std(σm_BCP)
εm_values_BCP, εm_std_BCP = extract_values_std(εm_BCP)
KIc_values_BCP, KIc_std_BCP = extract_values_std(KIc_BCP)
GIc_values_BCP, GIc_std_BCP = extract_values_std(GIc_BCP)

Et_values_CSR, Et_std_CSR = extract_values_std(Et_CSR)
σm_values_CSR, σm_std_CSR = extract_values_std(σm_CSR)
εm_values_CSR, εm_std_CSR = extract_values_std(εm_CSR)
KIc_values_CSR, KIc_std_CSR = extract_values_std(KIc_CSR)
GIc_values_CSR, GIc_std_CSR = extract_values_std(GIc_CSR)

# Extract values for BCP_CSR combination (different percentages)
percentages_BCP_CSR = [1, 2, 3] # Percentages are inferred from the combination e.g., 1BCP_1CSR, 2BCP_2CSR...
Et_values_BCP_CSR, Et_std_BCP_CSR = extract_values_std(Et_BCP_CSR)
σm_values_BCP_CSR, σm_std_BCP_CSR = extract_values_std(σm_BCP_CSR)
εm_values_BCP_CSR, εm_std_BCP_CSR = extract_values_std(εm_BCP_CSR)
KIc_values_BCP_CSR, KIc_std_BCP_CSR = extract_values_std(KIc_BCP_CSR)
GIc_values_BCP_CSR, GIc_std_BCP_CSR = extract_values_std(GIc_BCP_CSR)

# Base polymer MEP
Et_MEP = [(2880, 18)]
σm_MEP = [(68, 0.2)]
εm_MEP = [(4.3, 0.05)]
KIc_MEP = [(0.70, 0.1)]
GIc_MEP = [(0.15, 0.14)]

# Base polymer MEP values for extension
Et_values_MEP, Et_std_MEP = extract_values_std(Et_MEP)
σm_values_MEP, σm_std_MEP = extract_values_std(σm_MEP)
εm_values_MEP, εm_std_MEP = extract_values_std(εm_MEP)
KIc_values_MEP, KIc_std_MEP = extract_values_std(KIc_MEP)
GIc_values_MEP, GIc_std_MEP = extract_values_std(GIc_MEP)

# Percentage weight of the additive for BCP and CSR (since they have the same percentages)
percentages = [2, 4, 6, 8, 10, 12]

# Correcting the percentages for BCP_CSR combination
percentages_BCP_CSR = [2, 4, 6]

# Extending the BCP and CSR std dev lists with the base polymer MEP std dev data
Et_std_BCP_extended = Et_std_MEP + Et_std_BCP
Et_std_CSR_extended = Et_std_MEP + Et_std_CSR
Et_std_BCP_CSR_extended = Et_std_MEP[:len(percentages_BCP_CSR)] + Et_std_BCP_CSR

σm_std_BCP_extended = σm_std_MEP + σm_std_BCP
σm_std_CSR_extended = σm_std_MEP + σm_std_CSR
σm_std_BCP_CSR_extended = σm_std_MEP[:len(percentages_BCP_CSR)] + σm_std_BCP_CSR

εm_std_BCP_extended = εm_std_MEP + εm_std_BCP
εm_std_CSR_extended = εm_std_MEP + εm_std_CSR
εm_std_BCP_CSR_extended = εm_std_MEP[:len(percentages_BCP_CSR)] + εm_std_BCP_CSR

KIc_std_BCP_extended = KIc_std_MEP + KIc_std_BCP
KIc_std_CSR_extended = KIc_std_MEP + KIc_std_CSR
KIc_std_BCP_CSR_extended = KIc_std_MEP[:len(percentages_BCP_CSR)] + KIc_std_BCP_CSR

GIc_std_BCP_extended = GIc_std_MEP + GIc_std_BCP
GIc_std_CSR_extended = GIc_std_MEP + GIc_std_CSR
GIc_std_BCP_CSR_extended = GIc_std_MEP[:len(percentages_BCP_CSR)] + GIc_std_BCP_CSR

percentages_extended = [0] + percentages
percentages_BCP_CSR_extended = [0] + percentages_BCP_CSR

# Base polymer MEP values for extension
Et_values_MEP, Et_std_MEP = extract_values_std(Et_MEP)
σm_values_MEP, σm_std_MEP = extract_values_std(σm_MEP)
εm_values_MEP, εm_std_MEP = extract_values_std(εm_MEP)
KIc_values_MEP, KIc_std_MEP = extract_values_std(KIc_MEP)
GIc_values_MEP, GIc_std_MEP = extract_values_std(GIc_MEP)

# Extending the BCP and CSR lists with the base polymer MEP data
Et_values_BCP_extended = Et_values_MEP + Et_values_BCP
Et_values_CSR_extended = Et_values_MEP + Et_values_CSR
Et_values_BCP_CSR_extended = Et_values_MEP + Et_values_BCP_CSR

σm_values_BCP_extended = σm_values_MEP + σm_values_BCP
σm_values_CSR_extended = σm_values_MEP + σm_values_CSR
σm_values_BCP_CSR_extended = σm_values_MEP + σm_values_BCP_CSR

εm_values_BCP_extended = εm_values_MEP + εm_values_BCP
εm_values_CSR_extended = εm_values_MEP + εm_values_CSR
εm_values_BCP_CSR_extended = εm_values_MEP + εm_values_BCP_CSR

KIc_values_BCP_extended = KIc_values_MEP + KIc_values_BCP
KIc_values_CSR_extended = KIc_values_MEP + KIc_values_CSR
KIc_values_BCP_CSR_extended = KIc_values_MEP + KIc_values_BCP_CSR

GIc_values_BCP_extended = GIc_values_MEP + GIc_values_BCP
GIc_values_CSR_extended = GIc_values_MEP + GIc_values_CSR
GIc_values_BCP_CSR_extended = GIc_values_MEP + GIc_values_BCP_CSR


# Define a helper function to format polynomial coefficients into a string
def format_poly(coeffs):
    terms = []
    degree = len(coeffs) - 1
    for i, coeff in enumerate(coeffs):
        if degree - i > 1:
            terms.append(f"{coeff:.2f}x^{degree - i}")
        elif degree - i == 1:
            terms.append(f"{coeff:.2f}x")
        else:
            terms.append(f"{coeff:.2f}")
    return " + ".join(terms)

# Define a helper function to format polynomial coefficients into a string
def format_poly(coeffs):
    terms = []
    degree = len(coeffs) - 1
    for i, coeff in enumerate(coeffs):
        if degree - i > 1:
            terms.append(f"{coeff:.2f}x^{degree - i}")
        elif degree - i == 1:
            terms.append(f"{coeff:.2f}x")
        else:
            terms.append(f"{coeff:.2f}")
    return " + ".join(terms)

# Updated function with equation labels and data points in the legend
def plot_individual_property_quadratic_fit(x_values, y_values_BCP, y_std_BCP, y_values_CSR, y_std_CSR, y_values_BCP_CSR, y_std_BCP_CSR, y_label, label):
    plt.figure(figsize=(4.4, 3), dpi=200)
    
    # Define colors for clarity
    color_BCP = 'blue'
    color_CSR = 'green'
    color_BCP_CSR = 'red'
    
    # Fit and plot BCP quadratic line
    coeffs_BCP = np.polyfit(x_values, y_values_BCP, deg=2)
    poly_BCP = np.poly1d(coeffs_BCP)
    x_line = np.linspace(min(x_values), max(x_values), 100)
    plt.plot(x_line, poly_BCP(x_line), color=color_BCP)
    bcp_points = plt.errorbar(x_values, y_values_BCP, yerr=y_std_BCP, fmt='o', color=color_BCP, capsize=3, label=None)
    
    # Fit and plot CSR quadratic line
    coeffs_CSR = np.polyfit(x_values, y_values_CSR, deg=2)
    poly_CSR = np.poly1d(coeffs_CSR)
    plt.plot(x_line, poly_CSR(x_line), color=color_CSR)
    csr_points = plt.errorbar(x_values, y_values_CSR, yerr=y_std_CSR, fmt='s', color=color_CSR, capsize=3, label=None)
    
    # Fit and plot BCP_CSR quadratic line
    coeffs_BCP_CSR = np.polyfit(percentages_BCP_CSR_extended, y_values_BCP_CSR, deg=2)
    poly_BCP_CSR = np.poly1d(coeffs_BCP_CSR)
    x_line_BCP_CSR = np.linspace(min(percentages_BCP_CSR_extended), max(percentages_BCP_CSR_extended), 100)
    plt.plot(x_line_BCP_CSR, poly_BCP_CSR(x_line_BCP_CSR), color=color_BCP_CSR)
    bcp_csr_points = plt.errorbar(percentages_BCP_CSR_extended, y_values_BCP_CSR, yerr=y_std_BCP_CSR, fmt='^', color=color_BCP_CSR, capsize=3, label=None)
    
    # Creating custom legend handles
    legend_handles = [
        mlines.Line2D([], [], color=color_BCP, marker='o', markersize=5, label=f'BCP ({format_poly(coeffs_BCP)})'),
        mlines.Line2D([], [], color=color_CSR, marker='s', markersize=5, label=f'CSR ({format_poly(coeffs_CSR)})'),
        mlines.Line2D([], [], color=color_BCP_CSR, marker='^', markersize=5, label=f'BCP/CSR ({format_poly(coeffs_BCP_CSR)})')
    ]
    
    # Add label (e.g., "(a)") on the bottom left of the figure
    plt.text(-0.14, -0.2, label, transform=plt.gca().transAxes, fontsize=16, va='bottom', ha='left')

    plt.xlabel('Additive wt. %', fontsize=16)
    plt.ylabel(y_label, fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(handles=legend_handles, frameon=False, fontsize=9, loc='best')
    plt.grid(False)
    plt.tight_layout(pad=0.1)
    plt.show()

# Example of calling the updated plotting function for one of the properties
plot_individual_property_quadratic_fit(
    percentages_extended, Et_values_BCP_extended, Et_std_BCP_extended,
    Et_values_CSR_extended, Et_std_CSR_extended, Et_values_BCP_CSR_extended,
    Et_std_BCP_CSR_extended, '$E_{t}$ [MPa]', label='(a)')

# Tensile Strength ($\sigma_m$ [MPa])
plot_individual_property_quadratic_fit(
    percentages_extended, σm_values_BCP_extended, σm_std_BCP_extended,
    σm_values_CSR_extended, σm_std_CSR_extended, σm_values_BCP_CSR_extended,
    σm_std_BCP_CSR_extended, '$\sigma_{m}$ [MPa]', label='(b)')

# Elongation at Break ($\epsilon_m$ [%])
plot_individual_property_quadratic_fit(
    percentages_extended, εm_values_BCP_extended, εm_std_BCP_extended,
    εm_values_CSR_extended, εm_std_CSR_extended, εm_values_BCP_CSR_extended,
    εm_std_BCP_CSR_extended, '$\epsilon_{m}$ [%]', label='(c)')

# Fracture Toughness ($K_{Ic}$ [MPa.$m^{1/2}$])
plot_individual_property_quadratic_fit(
    percentages_extended, KIc_values_BCP_extended, KIc_std_BCP_extended,
    KIc_values_CSR_extended, KIc_std_CSR_extended, KIc_values_BCP_CSR_extended,
    KIc_std_BCP_CSR_extended, '$K_{Ic}$ [MPa.$m^{1/2}$]', label='(d)')

# Fracture Energy ($G_{Ic}$ [kJ/$m^2$])
plot_individual_property_quadratic_fit(
    percentages_extended, GIc_values_BCP_extended, GIc_std_BCP_extended,
    GIc_values_CSR_extended, GIc_std_CSR_extended, GIc_values_BCP_CSR_extended,
    GIc_std_BCP_CSR_extended, '$G_{Ic}$ [kJ/$m^2$]', label='(e)')