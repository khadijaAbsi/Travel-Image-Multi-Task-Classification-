import csv

# Valid values for Weather and Time of Day
valid_weather = {"Sunny", "Rainy", "Cloudy", "Snowy", "Not Clear"}
valid_time_of_day = {"Morning", "Afternoon", "Evening"}

input_file = "links_weather_time.csv"   # Input CSV file
output_file = "weather_time_clean.csv"  # Output CSV file

# Required columns (ONLY related to weather and time)
required_columns = ["Image URL", "Weather", "Time of Day"]

problem_lines = []

with open(input_file, newline='', encoding='utf-8') as csvfile:
    reader = list(csv.DictReader(csvfile))
    fixed_rows = []

    for i, row in enumerate(reader, start=2):  # Header is line 1
        row_fixed = {
            "Image URL": row.get("Image URL", "").strip(),
            "Weather": row.get("Weather", "").strip(),
            "Time of Day": row.get("Time of Day", "").strip()
        }

        problems = []

        # Validate Weather
        if row_fixed["Weather"] not in valid_weather:
            problems.append("Weather")

        # Validate Time of Day
        if row_fixed["Time of Day"] not in valid_time_of_day:
            problems.append("Time of Day")

        if problems:
            problem_lines.append((i, problems))

        fixed_rows.append(row_fixed)

# Write cleaned CSV file
with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=required_columns)
    writer.writeheader()
    writer.writerows(fixed_rows)

# Print lines with issues
if problem_lines:
    print("Found issues in the following lines:")
    for line_num, issues in problem_lines:
        print(f"Line {line_num}: {', '.join(issues)}")
else:
    print("No issues found!")
