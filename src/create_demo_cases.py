import pandas as pd
import os

# Original synthetic cases
input_file = "data/synthetic/missing_person_cases.csv"

# New small demo file
output_file = "data/synthetic/missing_person_cases_demo.csv"

print("Loading original case dataset...")

df = pd.read_csv(input_file)

print("Total original cases:", len(df))

# ---------------------------------------------------------
# Select representative cases from each target area
# ---------------------------------------------------------

if "target_area" in df.columns:

    areas = df["target_area"].dropna().unique()

    selected_parts = []

    # Try to select 5 cases from each area
    for area in areas:

        area_df = df[df["target_area"] == area]

        n = min(5, len(area_df))

        if n > 0:
            selected_parts.append(
                area_df.sample(
                    n=n,
                    random_state=42
                )
            )

    demo_df = pd.concat(
        selected_parts,
        ignore_index=True
    )

else:

    # Fallback if target_area is not available
    demo_df = df.sample(
        n=min(20, len(df)),
        random_state=42
    ).reset_index(drop=True)


# ---------------------------------------------------------
# Keep only 20 cases
# ---------------------------------------------------------

demo_df = demo_df.head(20).copy()

# ---------------------------------------------------------
# Create clean demo case IDs
# ---------------------------------------------------------

demo_df.insert(
    0,
    "demo_case_id",
    [
        f"MP-{i:05d}"
        for i in range(1, len(demo_df) + 1)
    ]
)

# Keep original case ID if available
if "case_id" in demo_df.columns:

    demo_df.rename(
        columns={
            "case_id": "original_case_id"
        },
        inplace=True
    )


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

demo_df.to_csv(
    output_file,
    index=False
)

print()
print("====================================")
print("DEMO CASES CREATED SUCCESSFULLY")
print("====================================")
print("Number of demo cases:", len(demo_df))
print("Saved at:", output_file)
print()

print("Demo Case IDs:")

print(
    demo_df["demo_case_id"].tolist()
)

if "target_area" in demo_df.columns:

    print()
    print("Cases by Target Area:")

    print(
        demo_df["target_area"]
        .value_counts()
    )