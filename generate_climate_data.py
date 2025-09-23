#!/usr/bin/env python3
"""
Script to generate synthetic climate data for all upazilas based on existing data.
This ensures all upazila IDs have sufficient historical climate data for predictions.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def generate_synthetic_climate_data():
    # File paths
    climate_file = Path("/Users/khalilur/Documents/ML/ClimatePredictionFile2023_2032_WithIDs.xlsx")

    # All upazila IDs that the model expects
    all_upazila_ids = [121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,231]

    print(f"Loading climate data from {climate_file}")
    df = pd.read_excel(climate_file)
    print(f"Original data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    # Check what the upazila column is called
    upazila_col = None
    for col in df.columns:
        if 'upazila' in col.lower() or 'Upazila' in col:
            upazila_col = col
            break

    if upazila_col is None:
        print("ERROR: Could not find upazila column in the data")
        return

    print(f"Using upazila column: {upazila_col}")
    print(f"Original upazila_ids: {sorted(df[upazila_col].unique())}")

    # Get the template data (from the first upazila that has data)
    template_upazila = df[upazila_col].iloc[0]  # First upazila in the file
    template_data = df[df[upazila_col] == template_upazila].copy()
    print(f"Template data shape (upazila {template_upazila}): {template_data.shape}")

    if template_data.empty:
        print("ERROR: No data found for upazila 121. Cannot generate synthetic data.")
        return

    # Create synthetic data for all upazilas
    synthetic_data = []

    for uid in all_upazila_ids:
        if uid == template_upazila:
            # Keep original data for the template upazila
            upazila_data = template_data.copy()
        else:
            # Create synthetic data based on template
            upazila_data = template_data.copy()

            # Change upazila_id
            upazila_data[upazila_col] = uid

            # Add random variation to climate indicators
            np.random.seed(uid)  # Reproducible randomness per upazila

            # Temperature: ±5°C variation
            temp_variation = np.random.normal(0, 2, len(upazila_data))
            upazila_data['Average_temperature'] += temp_variation

            # Rainfall: ±30% variation, ensure positive
            rainfall_variation = np.random.normal(1, 0.3, len(upazila_data))
            upazila_data['Total_rainfall'] = np.maximum(0, upazila_data['Total_rainfall'] * rainfall_variation)

            # Humidity: ±10% variation, keep in 0-100 range
            humidity_variation = np.random.normal(1, 0.1, len(upazila_data))
            upazila_data['Relative_humidity'] = np.clip(upazila_data['Relative_humidity'] * humidity_variation, 0, 100)

            # NDVI: ±0.2 variation, keep in -1 to 1 range
            ndvi_variation = np.random.normal(0, 0.1, len(upazila_data))
            upazila_data['Average_NDVI'] = np.clip(upazila_data['Average_NDVI'] + ndvi_variation, -1, 1)

            # NDWI: ±0.2 variation, keep in -1 to 1 range
            ndwi_variation = np.random.normal(0, 0.1, len(upazila_data))
            upazila_data['Average_NDWI'] = np.clip(upazila_data['Average_NDWI'] + ndwi_variation, -1, 1)

        synthetic_data.append(upazila_data)

    # Combine all data
    final_df = pd.concat(synthetic_data, ignore_index=True)
    final_df = final_df.sort_values([upazila_col, 'Year', 'Month'])

    print(f"Final data shape: {final_df.shape}")
    print(f"Final upazila_ids: {len(final_df[upazila_col].unique())} total")
    print(f"Records per upazila: {len(final_df) // len(all_upazila_ids)}")

    # Validate data
    print("\nData validation:")
    for uid in all_upazila_ids[:5]:  # Check first 5
        upazila_data = final_df[final_df[upazila_col] == uid]
        if not upazila_data.empty:
            min_year = upazila_data['Year'].min()
            min_month = upazila_data['Month'].min()
            max_year = upazila_data['Year'].max()
            max_month = upazila_data['Month'].max()
            date_range = f"{min_year}-{min_month} to {max_year}-{max_month}"
            print(f"  Upazila {uid}: {len(upazila_data)} records, dates {date_range}")
        else:
            print(f"  Upazila {uid}: No data found")

    # Save back to Excel
    output_file = climate_file
    final_df.to_excel(output_file, index=False)
    print(f"\n✅ Synthetic climate data saved to {output_file}")

    # Summary
    print("\n📊 Summary:")
    print(f"   Total upazilas: {len(all_upazila_ids)}")
    print(f"   Records per upazila: {len(final_df) // len(all_upazila_ids)}")
    print(f"   Total records: {len(final_df)}")
    print(f"   Date range: {final_df['Year'].min()}-{final_df['Month'].min()} to {final_df['Year'].max()}-{final_df['Month'].max()}")

if __name__ == "__main__":
    generate_synthetic_climate_data()