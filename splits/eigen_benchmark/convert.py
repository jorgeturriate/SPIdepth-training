input_file = "test_files.txt"
output_file = "test_files_mod.txt"

with open(input_file, "r") as infile, open(output_file, "w") as outfile:
    for line in infile:
        parts = line.strip().split()
        path, frame, side = parts

        # Extract the date from the folder name
        # Example: "data_depth_annotated/val/2011_09_26_drive_0002_sync"
        date = path.split("/")[-1].split("_drive")[0]

        # Frame number padded to 10 digits
        frame_padded = f"{int(frame):010d}"

        # New path format
        new_line = f"{date}/{path.split('/')[-1]} {frame_padded} {side}\n"
        outfile.write(new_line)
