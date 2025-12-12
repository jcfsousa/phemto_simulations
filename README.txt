#################   PHEMTO SIMULATION   #################

Megalib Folder organization:
    .
    ├── analysis_overall
    ├── archive
    │   ├── v1.0
    │   └── v1.1
    ├── detectors
    │   └── geometry
    ├── docs
    ├── instruments
    ├── notes
    │   └── miguel_help_scripts
    ├── run
    └── sources
        ├── spectrum
        └── simTra_files

    ./analysis_overall
        - Use this folder to creat data analysis scripts taking into account the output .tra files from megalib.

    ./archive
        - To archive old data (milestones).

    ./megalib/detectors/
        -Folder dedidated to .det files. Here we can specify the detector energy resolution, spatial resolution.

    ./megalib/detectors/geometry/
        -Folder dedicated to the individual detectors .geo characteristics (material, pixel, detector shape). The .det files use the .geo to make the each individual detector.

    ./megalib/instruments/
        -Folder dedicated to .geo.setup files. Here we can distribute the .det into the mother volume.

    ./run
        - Python3 scrips to produce bash commands to run cosima and revan. The python3 scripts expects the ./sources folder to be ../ from the python3 script. 
        - The prep_cosima.py script creats the .sources files (modify according to needs), writes the .souce at ../sources. The prep_revan.py creats the revan .sh commands to run revan taking into account the sources produced (at this stage the script has to be prepared in a similar way to the prep_cosima.py for the sources to be consistent).
        - The prep_run.py prepares a .sh to run cosima then revan for each configuration.

    ./sources
        - .sources to simulate that megalib uses for the simulation. It outputs .sim and .tra files to the directory ./sources/simTra_files

    ./sources/spectrum
        - to put some spectra data from either background, real sources (etc) to be included on the .source files
