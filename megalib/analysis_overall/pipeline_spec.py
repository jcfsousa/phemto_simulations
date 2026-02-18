import time
import subprocess
import os
## ------------------------------------------------------------
# Change Parameters

CdTe_matrix_size = [4]
dists = [1.5]
output_path = '/local/home/jf285468/documents/phd/phemto/phemto_simulations/megalib/sources/simTra_files'
sources_path = '/local/home/jf285468/documents/phd/phemto/phemto_simulations/megalib/sources'
SourceName = 'CollimatedBeam' #Bellow i'll use the nonpol for spectra analysis

config_lst = []
for matrix_size in CdTe_matrix_size:
    for dist in dists:
        config = f"config{matrix_size}x{matrix_size}_{dist}cm"
        config_lst.append(config)

# Energy list
#Log_E=[4,8,15,30,50,80,100,120,150,200,250,300,350,400,500,600,700]
Log_E=[4, 10, 30, 50, 100, 150, 200, 250, 300, 350, 400]




# Prep parse.py 
# Prep pre_process.py 
# Prep specMaker.py 

output_parser = '/local/home/jf285468/documents/phd/phemto/phemto_simulations/megalib/analysis_overall/polarimetry_data'
base_path_analysis_overall = '/local/home/jf285468/documents/phd/phemto/phemto_simulations/megalib/analysis_overall'

with open(f"./pipeline_spec_tmp.sh", mode='w') as f:
    f.write(f'source {base_path_analysis_overall}/env-phemto/bin/activate ;') # Activate python3 virtual environment
    for config in config_lst:
            for myene in Log_E:
                    ###### Parsing .tra -> .t3pa ######
                    # Non-Polarized source
                    tra_gz_filePathNonPol=f'{output_path}/{SourceName}NonPol{myene}keV_{config}.inc1.id1.tra.gz'
                    runNonPol_name = f"{SourceName}NonPol{myene}keV_{config}"
                    output_path_configNonPol = f"{output_parser}/{runNonPol_name}"
                    f.write(f'python3 {base_path_analysis_overall}/parse.py -f {tra_gz_filePathNonPol} -o {output_path_configNonPol};') # Non-Polarized Source

                    f.write(f'python3 {base_path_analysis_overall}/polarimetry/pre_process.py -i {output_path_configNonPol};') # Pre_process, single, double, mult, 

                    f.write(f'python3 {base_path_analysis_overall}/polarimetry/specMaker.py -i {output_path_configNonPol} -o {output_path_configNonPol};')

                    f.write(f'cp {output_path_configNonPol}/parquet/masked/spectra/*.txt /local/home/jf285468/documents/phd/phemto/phemto_simulations/results/megalib_v2.1/input_for_rmf/;')



time.sleep(1)

# Run the pipeline_spec.sh script
subprocess.run(['bash', './pipeline_spec_tmp.sh'])

os.remove('./pipeline_spec_tmp.sh') # remove tmp file

