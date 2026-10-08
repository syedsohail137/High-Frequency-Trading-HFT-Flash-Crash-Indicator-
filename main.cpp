#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <string>
#include "QuantResonanceEngine.hpp"

int main() {
    const std::string input_file = "input.txt";
    const std::string output_file = "report.txt";
    
    std::ifstream infile(input_file);
    if (!infile.is_open()) {
        std::cerr << "CRITICAL ERROR: Unable to locate input file: " << input_file << "\n";
        return 1;
    }

    std::ofstream outfile(output_file);
    if (!outfile.is_open()) {
        std::cerr << "CRITICAL ERROR: Cannot open output report path: " << output_file << "\n";
        return 1;
    }

    QuantResonanceEngine<256> engine;
    double running_variance = 0.0001; 
    double sampling_rate = 1.0;       
    int forecast_horizon = 120;       
    
    // Legacy thresholds kept for baseline comparisons
    double legacy_mu_limit = 2.50;
    double legacy_hurst_limit = 0.55;

    outfile << "========================================================================================\n";
    outfile << "      QUANT NON-LINEAR RESONANCE ENGINE (T+1 OPTIMIZED PREDICTIVE FORECASTER)          \n";
    outfile << "========================================================================================\n";
    outfile << "Step\tReturn\t\tGARCH_Var\tRes_w0\t\tMu_VDP\t\tHurst\tT+1_ALERT\tT+N_Collapse\n";
    outfile << "----------------------------------------------------------------------------------------\n";

    std::string line;
    int step_counter = 0;

    while (std::getline(infile, line)) {
        if (line.empty() || line[0] == '#') continue; 

        std::stringstream ss(line);
        double asset_return;
        if (!(ss >> asset_return)) continue;

        step_counter++;
        running_variance = engine.step_garch(asset_return, running_variance);
        
        // Pass tracking matrix context into analysis pipeline
        MarketMetrics metrics = engine.analyze_resonance(sampling_rate, legacy_mu_limit, legacy_hurst_limit, forecast_horizon);

        outfile << step_counter << "\t" 
                << std::fixed << std::setprecision(6) << asset_return << "\t"
                << running_variance << "\t"
                << metrics.w_0 << "\t"
                << metrics.structural_parameter << "\t"
                << metrics.hurst_exponent << "\t";
        
        // Print out the T+1 look-ahead warning flag status
        if (metrics.t_plus_1_warning) {
            outfile << "WARNING_T+1\t";
        } else {
            outfile << "CLEAR\t\t";
        }

        // Output legacy cascade integration tracking results
        if (metrics.predicted_time_to_collapse > 0) {
            outfile << "CRITICAL T+" << std::setprecision(2) << metrics.predicted_time_to_collapse << "\n";
        } else {
            outfile << "STABLE\n";
        }
    }

    std::cout << "Engine Execution Cycle Finalized successfully with T+1 Forward Shock Analytics.\n";
    return 0;
}
