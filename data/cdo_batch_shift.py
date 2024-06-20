import subprocess
import os

def create_shifted_file(input_file, output_dir):
	filename_segs = filename.split(".")[0:-1]
	filename_segs.append("shifted")
	filename_segs.append("nc")
	new_filename = ".".join(filename_segs)
	cdo_command = ["cdo","sellonlatbox,-180,180,-90,90",input_file,f"{output_dir}/{new_filename}"]
	subprocess.run(cdo_command)

unshifted_daily_temp_dir = "temp/daily/unshifted"
shifted_daily_temp_dir = "temp/daily/shifted"

unshifted_daily_precip_dir = "precip/daily/unshifted"
shifted_daily_precip_dir = "precip/daily/shifted"

unshifted_daily_humidity_dir = "humidity/daily/unshifted"
shifted_daily_humidity_dir = "humidity/daily/shifted"

unshifted_monthly_temp_dir = "temp/monthly/unshifted"
shifted_monthly_temp_dir = "temp/monthly/shifted"

unshifted_monthly_precip_dir = "precip/monthly/unshifted"
shifted_monthly_precip_dir = "precip/monthly/shifted"

unshifted_monthly_humidity_dir = "humidity/monthly/unshifted"
shifted_monthly_humidity_dir = "humidity/monthly/shifted"

for filename in os.listdir(unshifted_daily_temp_dir):
	create_shifted_file(f"{unshifted_daily_temp_dir}/{filename}",shifted_daily_temp_dir)
for filename in os.listdir(unshifted_daily_precip_dir):
	create_shifted_file(f"{unshifted_daily_precip_dir}/{filename}",shifted_daily_precip_dir)
for filename in os.listdir(unshifted_daily_humidity_dir):
	create_shifted_file(f"{unshifted_daily_humidity_dir}/{filename}",shifted_daily_humidity_dir)
for filename in os.listdir(unshifted_monthly_temp_dir):
	create_shifted_file(f"{unshifted_monthly_temp_dir}/{filename}",shifted_monthly_temp_dir)
for filename in os.listdir(unshifted_monthly_precip_dir):
	create_shifted_file(f"{unshifted_monthly_precip_dir}/{filename}",shifted_monthly_precip_dir)
for filename in os.listdir(unshifted_monthly_humidity_dir):
	create_shifted_file(f"{unshifted_monthly_humidity_dir}/{filename}",shifted_monthly_humidity_dir)
	
	