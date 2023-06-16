import csv

# Open the .txt file

with open("wings_of_fire.txt", "r") as f:
    text = f.read()

# Write the text to a .csv file

with open("wings_of_fire.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(text.splitlines())