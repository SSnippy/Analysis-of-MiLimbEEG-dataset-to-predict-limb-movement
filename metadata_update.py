import os
import pandas as pd

# =========================
# CONFIGURATION
# =========================

PROJECT_ROOT = "/home/snippy/Desktop/Projects/Milimb_eeg"
METADATA_PATH = os.path.join(PROJECT_ROOT, "datapoints", "metadata.xlsx")

# =========================
# LOAD METADATA
# =========================

df = pd.read_excel(METADATA_PATH)

if "local_url" not in df.columns:
    raise ValueError("Column 'local_url' not found in metadata.xlsx")

# =========================
# BUILD FILTERED URL COLUMN
# =========================

def build_filtered_url(relative_raw_path):
    """
    Converts a relative raw EEG path to its corresponding
    relative FIR-filtered path.
    """
    parts = relative_raw_path.split(os.sep)

    # Expected:
    # ['datapoints', 'raw', 'Sx', 'filename.csv']
    if len(parts) < 4:
        return None

    datapoints, _, patient, filename = parts[:4]

    return os.path.join(
        datapoints,
        "fir",
        f"f_{patient}",
        f"f_{filename}"
    )

filtered_urls = df["local_url"].apply(build_filtered_url)

# =========================
# INSERT COLUMN AFTER local_url
# =========================

local_url_index = df.columns.get_loc("local_url")
df.insert(local_url_index + 1, "filtered_url", filtered_urls)

# =========================
# CORRESPONDENCE CHECK (NO COLUMN CREATED)
# =========================

def check_correspondence(local_url, filtered_url):
    """
    Verifies that:
    datapoints/raw/Sx/<filename>
    ↔
    datapoints/fir/f_Sx/f_<filename>
    """
    try:
        # From raw path
        raw_tail = local_url.split(os.sep + "raw" + os.sep)[1]
        raw_filename = os.path.basename(raw_tail)

        # From filtered path
        fir_tail = filtered_url.split(os.sep + "fir" + os.sep)[1]
        fir_filename = os.path.basename(fir_tail)

        if not fir_filename.startswith("f_"):
            return False

        fir_filename = fir_filename[2:]

        return raw_filename == fir_filename

    except Exception:
        return False

# Run validation
results = df.apply(
    lambda row: check_correspondence(row["local_url"], row["filtered_url"]),
    axis=1
)

# =========================
# SAVE BACK TO EXCEL
# =========================

df.to_excel(METADATA_PATH, index=False)

# =========================
# SUMMARY REPORT
# =========================

total = len(results)
matched = results.sum()
mismatched = total - matched

print("metadata.xlsx updated")
print(f"Total rows checked : {total}")
print(f"Valid mappings    : {matched}")
print(f"Invalid mappings  : {mismatched}")

if mismatched > 0:
    print("\nFirst 10 mismatches:")
    print(
        df.loc[~results, ["local_url", "filtered_url"]]
        .head(10)
        .to_string(index=False)
    )
