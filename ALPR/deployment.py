from data_pipeline import Pipeline

# Instantiate a pipleline class
pipeline=Pipeline()

frames = pipeline.stream_to_frame('udp://127.0.0.1:23002', 3840, 2160)
#frames = pipeline.stream_to_frame('LicensePlateReaderSample_4k.mov', 3840, 2160)

plate_count = {}
frames_processed = 0

for frame in frames:
    #print(frame)
    license_plate_image = pipeline.license_plate(frame)
    if license_plate_image is not None:
        plate_number = pipeline.read_cropped_image(license_plate_image)
        if plate_number not in ["Invalid Characters", "Invalid Length", "Nothing Detected"]:
            plate_count[plate_number] = plate_count.get(plate_number, 0) + 1
        print(plate_number)
    frames_processed += 1
    
# Print number of unique plate numbers and their frequencies
print("Number of unique plate numbers:", len(plate_count))

for plate_number, count in plate_count.items():
    print(f"Plate number {plate_number} detected {count} times.")

# Print message indicating completion
print("The video has been done analysing.")    
    

# Outputting results to a text file
output_file = "results.txt"

with open(output_file, "w") as f:
    # Write the number of unique plate numbers and their frequencies
    f.write("Number of unique plate numbers: {}\n".format(len(plate_count)))
    for plate_number, count in plate_count.items():
        f.write(f"Plate number {plate_number} detected {count} times.\n")

    # Write the completion message
    f.write("The video has been done analysing.\n")

# Print a message indicating that the results have been saved
print("Results have been saved to", output_file)

