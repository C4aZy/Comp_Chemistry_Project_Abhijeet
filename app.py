from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from io import BytesIO
import base64
import re
from scipy.interpolate import interp1d, CubicSpline
from scipy.signal import find_peaks, argrelextrema
from scipy.optimize import curve_fit

app = Flask(__name__)

# Store data in memory
data_store = []

# File upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/add_point', methods=['POST'])
def add_point():
    """Add a new data point"""
    try:
        data = request.json
        angle = float(data['angle'])
        energy = float(data['energy'])
        
        data_store.append({'angle': angle, 'energy': energy})
        data_store.sort(key=lambda x: x['angle'])
        
        return jsonify({'status': 'success', 'data': data_store})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/upload_file', methods=['POST'])
def upload_file():
    """Upload and parse computational chemistry output files"""
    try:
        if 'file' not in request.files:
            return jsonify({'status': 'error', 'message': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'status': 'error', 'message': 'No file selected'}), 400
        
        filename = file.filename.lower()
        content = file.read().decode('utf-8', errors='ignore')
        
        # Parse different file formats
        if filename.endswith('.log') or filename.endswith('.out'):
            # Gaussian/ORCA format
            parsed_data = parse_gaussian_orca(content)
        elif filename.endswith('.csv'):
            # CSV format
            parsed_data = parse_csv(content)
        elif filename.endswith('.txt'):
            # Generic text format
            parsed_data = parse_generic(content)
        else:
            return jsonify({'status': 'error', 'message': 'Unsupported file format'}), 400
        
        if parsed_data:
            data_store.clear()
            data_store.extend(parsed_data)
            data_store.sort(key=lambda x: x['angle'])
            return jsonify({'status': 'success', 'data': data_store, 'count': len(parsed_data)})
        else:
            return jsonify({'status': 'error', 'message': 'Could not parse file'}), 400
            
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

def parse_gaussian_orca(content):
    """Parse Gaussian or ORCA output files"""
    data = []
    lines = content.split('\n')
    
    # Try to find scan data
    for i, line in enumerate(lines):
        # Gaussian: "Scan" keyword
        if 'Summary of the potential surface scan:' in line or 'Scan' in line:
            # Look for angle and energy patterns
            for j in range(i, min(i+200, len(lines))):
                match = re.search(r'(\d+\.?\d*)\s+(-?\d+\.?\d+)', lines[j])
                if match:
                    angle = float(match.group(1))
                    energy = float(match.group(2))
                    data.append({'angle': angle, 'energy': energy})
        
        # ORCA: Relaxed surface scan
        if 'RELAXED SURFACE SCAN' in line:
            for j in range(i, min(i+200, len(lines))):
                match = re.search(r'(\d+\.?\d*)\s+(-?\d+\.?\d+)', lines[j])
                if match:
                    angle = float(match.group(1))
                    energy = float(match.group(2))
                    data.append({'angle': angle, 'energy': energy})
    
    return data

def parse_csv(content):
    """Parse CSV file"""
    data = []
    lines = content.strip().split('\n')
    
    # Skip header if present
    start_idx = 1 if 'angle' in lines[0].lower() else 0
    
    for line in lines[start_idx:]:
        parts = line.split(',')
        if len(parts) >= 2:
            try:
                angle = float(parts[0].strip())
                energy = float(parts[1].strip())
                data.append({'angle': angle, 'energy': energy})
            except ValueError:
                continue
    
    return data

def parse_generic(content):
    """Parse generic text format (space or tab separated)"""
    data = []
    lines = content.strip().split('\n')
    
    for line in lines:
        # Skip comments and empty lines
        if line.strip().startswith('#') or not line.strip():
            continue
        
        parts = re.split(r'\s+', line.strip())
        if len(parts) >= 2:
            try:
                angle = float(parts[0])
                energy = float(parts[1])
                data.append({'angle': angle, 'energy': energy})
            except ValueError:
                continue
    
    return data

@app.route('/get_data', methods=['GET'])
def get_data():
    """Get all data points"""
    return jsonify({'data': data_store})

@app.route('/delete_point/<int:index>', methods=['DELETE'])
def delete_point(index):
    """Delete a specific data point"""
    try:
        if 0 <= index < len(data_store):
            data_store.pop(index)
            return jsonify({'status': 'success', 'data': data_store})
        return jsonify({'status': 'error', 'message': 'Invalid index'}), 400
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/clear_data', methods=['POST'])
def clear_data():
    """Clear all data"""
    data_store.clear()
    return jsonify({'status': 'success'})

@app.route('/download_csv', methods=['GET'])
def download_csv():
    """Generate and download CSV file"""
    if not data_store:
        return jsonify({'status': 'error', 'message': 'No data available'}), 400
    
    df = pd.DataFrame(data_store)
    output = BytesIO()
    df.to_csv(output, index=False)
    output.seek(0)
    
    return send_file(
        output,
        mimetype='text/csv',
        as_attachment=True,
        download_name='pes_data.csv'
    )

@app.route('/advanced_analysis', methods=['GET'])
def advanced_analysis():
    """Perform advanced PES analysis including Conformational Stability"""
    if len(data_store) < 3:
        return jsonify({'status': 'error', 'message': 'Need at least 3 data points'}), 400
    
    try:
        df = pd.DataFrame(data_store)
        angles = df['angle'].values
        energies = df['energy'].values
        
        # Normalize energies (relative to minimum)
        min_energy = np.min(energies)
        rel_energies = energies - min_energy
        
        # Find minima and maxima
        minima_idx = argrelextrema(energies, np.less)[0]
        maxima_idx = argrelextrema(energies, np.greater)[0]
        
        minima = [{'angle': float(angles[i]), 'energy': float(energies[i]), 
                   'relative_energy': float(rel_energies[i])} for i in minima_idx]
        maxima = [{'angle': float(angles[i]), 'energy': float(energies[i]),
                   'relative_energy': float(rel_energies[i])} for i in maxima_idx]
        
        # Global minimum
        global_min_idx = np.argmin(energies)
        global_min = {
            'angle': float(angles[global_min_idx]),
            'energy': float(energies[global_min_idx])
        }
        
        # Energy barriers
        barriers = []
        for maximum in maxima:
            barrier = maximum['energy'] - global_min['energy']
            barriers.append({
                'from_angle': global_min['angle'],
                'to_angle': maximum['angle'],
                'barrier_height': float(barrier),
                'barrier_kcal': float(barrier * 0.239006)
            })
        
        # Fourier fit
        if len(angles) >= 6:
            fourier_params = fit_fourier_series(angles, rel_energies)
        else:
            fourier_params = None
        
        # Interpolated curve
        if len(angles) >= 4:
            spline = CubicSpline(angles, energies)
            interp_angles = np.linspace(angles.min(), angles.max(), 200)
            interp_energies = spline(interp_angles)
        else:
            interp_angles = angles
            interp_energies = energies

        # ---------------------------------------------------------
        # 🔵 CONFORMATIONAL STABILITY SECTION (NEW)
        # ---------------------------------------------------------

        # Most and least stable conformations
        most_stable_idx = np.argmin(energies)
        least_stable_idx = np.argmax(energies)

        stability_range = energies[least_stable_idx] - energies[most_stable_idx]

        # Boltzmann populations at 298 K
        R = 0.008314  # kJ/mol·K
        T = 298
        boltz_factors = np.exp(-(energies - energies[most_stable_idx]) / (R * T))
        boltz_population = boltz_factors / np.sum(boltz_factors)

        conformational_stability = {
            'most_stable_angle': float(angles[most_stable_idx]),
            'most_stable_energy': float(energies[most_stable_idx]),
            'least_stable_angle': float(angles[least_stable_idx]),
            'least_stable_energy': float(energies[least_stable_idx]),
            'stability_range_kj': float(stability_range),
            'stability_range_kcal': float(stability_range * 0.239006),
            'boltzmann_population': [float(x) for x in boltz_population]
        }

        # ---------------------------------------------------------

        return jsonify({
            'status': 'success',
            'global_minimum': global_min,
            'local_minima': minima,
            'transition_states': maxima,
            'energy_barriers': barriers,
            'fourier_params': fourier_params,
            'conformational_stability': conformational_stability,  # NEW BLOCK
            'interpolated': {
                'angles': interp_angles.tolist(),
                'energies': interp_energies.tolist()
            },
            'statistics': {
                'num_points': len(data_store),
                'energy_range': float(energies.max() - energies.min()),
                'angle_range': float(angles.max() - angles.min())
            }
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400


def fit_fourier_series(angles, energies, n_terms=3):
    """Fit PES to Fourier series: E = V0 + sum(Vn * (1 + cos(n*theta)))"""
    try:
        angles_rad = np.radians(angles)
        
        def fourier_func(theta, V0, V1, V2, V3):
            return V0 + V1*(1 + np.cos(theta)) + V2*(1 + np.cos(2*theta)) + V3*(1 + np.cos(3*theta))
        
        popt, _ = curve_fit(fourier_func, angles_rad, energies, maxfev=5000)
        
        return {
            'V0': float(popt[0]),
            'V1': float(popt[1]),
            'V2': float(popt[2]),
            'V3': float(popt[3])
        }
    except:
        return None

@app.route('/generate_advanced_plot', methods=['GET'])
def generate_advanced_plot():
    """Generate advanced matplotlib plot with annotations"""
    if not data_store:
        return jsonify({'status': 'error', 'message': 'No data available'}), 400
    
    try:
        df = pd.DataFrame(data_store)
        angles = df['angle'].values
        energies = df['energy'].values
        
        # Normalize energies
        min_energy = np.min(energies)
        rel_energies = energies - min_energy
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Main PES with annotations
        ax1.plot(angles, rel_energies, 'o-', linewidth=2, markersize=8, label='Calculated Points')
        
        # Interpolated curve
        if len(angles) >= 4:
            spline = CubicSpline(angles, rel_energies)
            interp_angles = np.linspace(angles.min(), angles.max(), 200)
            interp_energies = spline(interp_angles)
            ax1.plot(interp_angles, interp_energies, '--', alpha=0.5, label='Cubic Spline')
        
        # Find and mark minima and maxima
        minima_idx = argrelextrema(rel_energies, np.less)[0]
        maxima_idx = argrelextrema(rel_energies, np.greater)[0]
        
        if len(minima_idx) > 0:
            ax1.plot(angles[minima_idx], rel_energies[minima_idx], 'g^', 
                    markersize=12, label='Local Minima')
            for idx in minima_idx:
                ax1.annotate(f'{rel_energies[idx]:.2f}', 
                           (angles[idx], rel_energies[idx]),
                           xytext=(0, -15), textcoords='offset points',
                           ha='center', fontsize=9, color='green')
        
        if len(maxima_idx) > 0:
            ax1.plot(angles[maxima_idx], rel_energies[maxima_idx], 'rv', 
                    markersize=12, label='Transition States')
            for idx in maxima_idx:
                ax1.annotate(f'{rel_energies[idx]:.2f}', 
                           (angles[idx], rel_energies[idx]),
                           xytext=(0, 10), textcoords='offset points',
                           ha='center', fontsize=9, color='red')
        
        # Mark global minimum
        global_min_idx = np.argmin(rel_energies)
        ax1.plot(angles[global_min_idx], rel_energies[global_min_idx], 
                'b*', markersize=20, label='Global Minimum')
        
        ax1.set_xlabel('Angle (°)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Relative Energy (kJ/mol)', fontsize=12, fontweight='bold')
        ax1.set_title('Potential Energy Surface Analysis', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best')
        
        # Plot 2: Energy distribution
        ax2.hist(rel_energies, bins=min(20, len(rel_energies)), 
                edgecolor='black', alpha=0.7)
        ax2.set_xlabel('Relative Energy (kJ/mol)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax2.set_title('Energy Distribution', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Save to bytes
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
        img_buffer.seek(0)
        plt.close()
        
        img_base64 = base64.b64encode(img_buffer.read()).decode()
        
        return jsonify({
            'status': 'success',
            'image': f'data:image/png;base64,{img_base64}'
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/download_plot', methods=['GET'])
def download_plot():
    """Download the matplotlib plot as PNG"""
    if not data_store:
        return jsonify({'status': 'error', 'message': 'No data available'}), 400
    
    df = pd.DataFrame(data_store)
    
    plt.figure(figsize=(10, 6))
    plt.plot(df['angle'], df['energy'], marker='o', linestyle='-', linewidth=2, markersize=8)
    plt.xlabel('Angle (°)', fontsize=12)
    plt.ylabel('Energy (kJ/mol)', fontsize=12)
    plt.title('Potential Energy Surface (PES)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
    img_buffer.seek(0)
    plt.close()
    
    return send_file(
        img_buffer,
        mimetype='image/png',
        as_attachment=True,
        download_name='pes_plot.png'
    )

@app.route('/export_analysis_report', methods=['GET'])
def export_analysis_report():
    """Export comprehensive analysis report"""
    if not data_store:
        return jsonify({'status': 'error', 'message': 'No data available'}), 400
    
    try:
        # Get analysis data
        df = pd.DataFrame(data_store)
        angles = df['angle'].values
        energies = df['energy'].values
        min_energy = np.min(energies)
        rel_energies = energies - min_energy
        
        # Find critical points
        minima_idx = argrelextrema(energies, np.less)[0]
        maxima_idx = argrelextrema(energies, np.greater)[0]
        global_min_idx = np.argmin(energies)
        
        # Calculate Boltzmann populations for stability analysis
        R = 8.314  # J/(mol·K)
        T = 298.15  # K
        energies_J = energies * 1000
        rel_energies_J = energies_J - np.min(energies_J)
        boltzmann_factors = np.exp(-rel_energies_J / (R * T))
        partition_function = np.sum(boltzmann_factors)
        populations = (boltzmann_factors / partition_function) * 100
        
        # Create report
        report = "=" * 70 + "\n"
        report += "POTENTIAL ENERGY SURFACE ANALYSIS REPORT\n"
        report += "=" * 70 + "\n\n"
        
        report += f"Total Data Points: {len(data_store)}\n"
        report += f"Angle Range: {angles.min():.2f}° to {angles.max():.2f}°\n"
        report += f"Energy Range: {energies.max() - energies.min():.4f} kJ/mol\n\n"
        
        report += "-" * 70 + "\n"
        report += "GLOBAL MINIMUM\n"
        report += "-" * 70 + "\n"
        report += f"Angle: {angles[global_min_idx]:.2f}°\n"
        report += f"Energy: {energies[global_min_idx]:.6f} kJ/mol\n"
        report += f"Population at 298.15 K: {populations[global_min_idx]:.4f}%\n\n"
        
        if len(minima_idx) > 0:
            report += "-" * 70 + "\n"
            report += "LOCAL MINIMA\n"
            report += "-" * 70 + "\n"
            for i, idx in enumerate(minima_idx, 1):
                report += f"{i}. Angle: {angles[idx]:.2f}°, "
                report += f"Energy: {energies[idx]:.6f} kJ/mol, "
                report += f"Relative: {rel_energies[idx]:.4f} kJ/mol\n"
                report += f"   Population: {populations[idx]:.4f}%\n"
            report += "\n"
        
        if len(maxima_idx) > 0:
            report += "-" * 70 + "\n"
            report += "TRANSITION STATES (MAXIMA)\n"
            report += "-" * 70 + "\n"
            for i, idx in enumerate(maxima_idx, 1):
                barrier = energies[idx] - energies[global_min_idx]
                report += f"{i}. Angle: {angles[idx]:.2f}°, "
                report += f"Energy: {energies[idx]:.6f} kJ/mol\n"
                report += f"   Barrier from global min: {barrier:.4f} kJ/mol "
                report += f"({barrier * 0.239006:.4f} kcal/mol)\n"
            report += "\n"
        
        # Conformational stability section
        report += "-" * 70 + "\n"
        report += "CONFORMATIONAL STABILITY ANALYSIS (298.15 K)\n"
        report += "-" * 70 + "\n"
        significant_conformers = np.sum(populations > 1.0)
        report += f"Significant Conformers (>1% population): {significant_conformers}\n"
        report += f"Dominant Conformer Population: {populations[global_min_idx]:.4f}%\n\n"
        
        # Sort by population
        pop_indices = np.argsort(populations)[::-1]
        report += "Top Conformers by Population:\n"
        for i, idx in enumerate(pop_indices[:5], 1):  # Top 5
            if populations[idx] > 0.01:  # Only show if >0.01%
                report += f"{i}. Angle: {angles[idx]:.2f}°, "
                report += f"ΔE: {rel_energies[idx]:.4f} kJ/mol, "
                report += f"Population: {populations[idx]:.4f}%\n"
        report += "\n"
        
        report += "-" * 70 + "\n"
        report += "DATA TABLE\n"
        report += "-" * 70 + "\n"
        report += f"{'Angle (°)':<15}{'Energy (kJ/mol)':<20}{'Relative (kJ/mol)':<20}{'Population (%)':<15}\n"
        report += "-" * 70 + "\n"
        for i in range(len(angles)):
            report += f"{angles[i]:<15.2f}{energies[i]:<20.6f}{rel_energies[i]:<20.4f}{populations[i]:<15.4f}\n"
        
        # Return as text file
        output = BytesIO()
        output.write(report.encode('utf-8'))
        output.seek(0)
        
        return send_file(
            output,
            mimetype='text/plain',
            as_attachment=True,
            download_name='pes_analysis_report.txt'
        )
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/conformational_stability', methods=['GET'])
def conformational_stability():
    """Calculate conformational stability analysis including Boltzmann populations"""
    if len(data_store) < 2:
        return jsonify({'status': 'error', 'message': 'Need at least 2 data points'}), 400
    
    try:
        df = pd.DataFrame(data_store)
        angles = df['angle'].values
        energies = df['energy'].values
        
        # Constants
        R = 8.314  # J/(mol·K) - Gas constant
        T = 298.15  # K - Temperature (25°C)
        
        # Convert kJ/mol to J/mol
        energies_J = energies * 1000
        
        # Calculate relative energies from global minimum
        min_energy = np.min(energies_J)
        rel_energies_J = energies_J - min_energy
        
        # Calculate Boltzmann factors: exp(-ΔE/RT)
        boltzmann_factors = np.exp(-rel_energies_J / (R * T))
        
        # Calculate partition function (sum of all Boltzmann factors)
        partition_function = np.sum(boltzmann_factors)
        
        # Calculate populations (%)
        populations = (boltzmann_factors / partition_function) * 100
        
        # Find minima for conformer analysis
        minima_idx = argrelextrema(energies, np.less)[0]
        
        # Include global minimum if not already in minima_idx
        global_min_idx = np.argmin(energies)
        if global_min_idx not in minima_idx:
            minima_idx = np.append(minima_idx, global_min_idx)
            minima_idx = np.sort(minima_idx)
        
        # Conformer data
        conformers = []
        for i, idx in enumerate(minima_idx, 1):
            conformers.append({
                'conformer_id': i,
                'angle': float(angles[idx]),
                'energy': float(energies[idx]),
                'relative_energy_kJ': float((energies[idx] - energies[global_min_idx])),
                'relative_energy_kcal': float((energies[idx] - energies[global_min_idx]) * 0.239006),
                'population_percent': float(populations[idx]),
                'is_global_minimum': bool(idx == global_min_idx)
            })
        
        # Sort by population (highest first)
        conformers_by_pop = sorted(conformers, key=lambda x: x['population_percent'], reverse=True)
        
        # Calculate thermodynamic parameters
        # Free energy: G = E - T*S (simplified: G ≈ E for conformational analysis)
        # Entropy contribution from populations
        entropy_contrib = -R * np.sum(populations/100 * np.log(populations/100 + 1e-10))
        
        # Stability metrics
        stability_metrics = {
            'most_stable_conformer': conformers_by_pop[0]['conformer_id'],
            'most_stable_angle': conformers_by_pop[0]['angle'],
            'most_stable_population': conformers_by_pop[0]['population_percent'],
            'num_significant_conformers': int(np.sum(populations > 1.0)),  # Conformers with >1% population
            'temperature_K': T,
            'temperature_C': T - 273.15,
            'entropy_contribution': float(entropy_contrib)
        }
        
        # All points with populations
        all_points_data = []
        for i in range(len(angles)):
            all_points_data.append({
                'angle': float(angles[i]),
                'energy': float(energies[i]),
                'relative_energy_kJ': float(energies[i] - energies[global_min_idx]),
                'population_percent': float(populations[i])
            })
        
        return jsonify({
            'status': 'success',
            'conformers': conformers,
            'conformers_by_population': conformers_by_pop,
            'stability_metrics': stability_metrics,
            'all_points': all_points_data,
            'summary': {
                'global_minimum_angle': float(angles[global_min_idx]),
                'global_minimum_energy': float(energies[global_min_idx]),
                'num_conformers': len(conformers),
                'dominant_conformer_population': float(conformers_by_pop[0]['population_percent'])
            }
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/get_summary', methods=['GET'])
def get_summary():
    """Get lowest energy conformer summary"""
    if not data_store:
        return jsonify({'status': 'error', 'message': 'No data available'}), 400
    
    df = pd.DataFrame(data_store)
    min_idx = df['energy'].idxmin()
    
    return jsonify({
        'status': 'success',
        'lowest_energy': float(df['energy'][min_idx]),
        'lowest_angle': float(df['angle'][min_idx])
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)