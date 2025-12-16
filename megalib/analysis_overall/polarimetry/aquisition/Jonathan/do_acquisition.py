import pypixet
import os
import traceback
import sys
import time



# ascii file as output
FILETYPE = '.t3pa' 

# specify output file directory
#OUT_DIR = 'C:/Users/jonat/Documents/LIP Coimbra/PIXET/out-files/'
OUT_DIR = '/home/mariana/Documents/THOR-SR/Timepix3-Python-API/out-files/'
FILE_DIR_LOG = 'logfiles/'

if not os.path.exists(OUT_DIR):
    OUT_DIR = 'out-files/'
    FILE_DIR_LOG = OUT_DIR + FILE_DIR_LOG

def config():
    """Configures the acquisition parameters."""
    filename = str(input("Enter filename: "))
    actime = int(input("Enter total acquisition time in seconds: ")) 
    interval = int(input("Enter duration of each interval in seconds: "))
    bias = float(input("Enter bias voltage in volts: "))
    out_dir = OUT_DIR + filename + '/'
    filepath_log = FILE_DIR_LOG + filename + '_log.txt'
    return filename, actime, interval, bias, out_dir, filepath_log

def create_directory(file_dir):
    """Creates a directory if it doesn't exist."""
    os.makedirs(file_dir, exist_ok=True)

def create_logfile(filepath_log, filename, actime, interval, bias):
    create_directory(FILE_DIR_LOG)
    with open(filepath_log, 'w') as f:
        f.write('logfile created at ' + time.strftime('%Y-%m-%d %H:%M:%S') + '\n\n')
        f.write(f"filename: {filename} \n")
        f.write(f"total acquisition time: {actime} seconds\n")
        f.write(f"duration of each interval: {interval} seconds\n")
        f.write(f"bias input voltage: {bias} V\n\n")
        f.write('timestamp | temperature (°C) | bias set voltage (V) | bias sense voltage (V) | bias sense current (µA) \n')

def log_temp_bias(device, file):
    temp = round(device.temperature(), 3)
    bias_set = device.bias()
    bias_v = round(device.biasVoltageSense(), 3)
    bias_c = round(device.biasCurrentSense(), 5)
    file.write(f"{round(time.time(),1)} {temp} {bias_set} {bias_v} {bias_c}\n")
    print('Temperature:', temp, '°C')
    print('Bias set voltage:', bias_set, 'V')
    print('Bias sense voltage:', bias_v, 'V')
    print('Bias sense current:', bias_c, 'µA')

def get_file_name(sequence, out_dir, filename):
    if sequence == 0:
        return out_dir + filename + FILETYPE
    else:
        return out_dir + filename + '-' + str(sequence) + FILETYPE

def startup():
    pypixet.start()
    pixet = pypixet.pixet
    pixet.refreshDevices()
    try:
        devices = pixet.devicesByType(pixet.PX_DEVTYPE_TPX3)
        dev = devices[0]
    except Exception:
        sys.exit("No TPX3 device found.")
    print('Device', dev.fullName(), 'found.')
    return pixet, dev

def end(pixet):
    pixet.exitPixet()
    pypixet.exit()

def single_acquisition(pixet, dev, sequence, time, out_dir, filename):
    print("doAdvancedAcquisition...")
    rc = dev.doAdvancedAcquisition(1, time, pixet.PX_ACQTYPE_DATADRIVEN, pixet.PX_ACQMODE_NORMAL, pixet.PX_FTYPE_AUTODETECT, 0, get_file_name(sequence, out_dir, filename))
    print(" rc", rc, "(0 is OK)")
    if rc != 0:
    	raise Exception
    

def do_single_acquisition(pixet, dev, sequence, time, filepath_log, out_dir, filename):
    with open(filepath_log, 'a') as f:
        try:
            dev.doSensorRefresh()
            log_temp_bias(dev, f)
            single_acquisition(pixet, dev, sequence, time, out_dir, filename)
        except Exception as e:
                if e == KeyboardInterrupt:
                    print("Acquisition stopped by user.")
                    end(pixet)
                    sys.exit("KeyboardInterrupt")
                else:
                    print("Exception:", e)
                    traceback.print_exc()
                    f.write(f"Exception: {e} \n")

def do_acquisition():
    pixet, dev = startup()
    filename, actime, interval, bias, out_dir, filepath_log = config()

    dev.setBias(bias)
    dev.setOperationMode(pixet.PX_TPX3_OPM_TOT_NOTOA)

    create_directory(out_dir)
    create_logfile(filepath_log, filename, actime, interval, bias)

    if actime > interval:
    
        number_of_sequences, rest_time = divmod(actime, interval)
        
        if rest_time > 0:
            number_of_sequences += 1
        else:
            rest_time = interval
        for i in range(number_of_sequences-1):
            do_single_acquisition(pixet, dev, i+1, interval, filepath_log, out_dir, filename)
            print("Sequence", i+1, "of", number_of_sequences, "done.")
        do_single_acquisition(pixet, dev, number_of_sequences, rest_time, filepath_log,
        out_dir, filename)
        print("Sequence", number_of_sequences, "of", number_of_sequences, "done.") 
  
    else:
        do_single_acquisition(pixet, dev, 0, actime, filepath_log, out_dir, filename)
        print("Sequence 1 of 1 done.")

    end(pixet)

if __name__ == "__main__":
    do_acquisition()
