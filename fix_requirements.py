# fix_requirements.py
import os

# The exact list of libraries your app needs
requirements = """streamlit
pandas
numpy
scikit-learn
xgboost
"""

# Write to file with UTF-8 encoding (Linux compatible)
file_path = "requirements.txt"
with open(file_path, "w", encoding="utf-8") as f:
    f.write(requirements)

print(f"✅ Success! Created {file_path} with standard UTF-8 encoding.")
print("   Content:")
print(requirements)