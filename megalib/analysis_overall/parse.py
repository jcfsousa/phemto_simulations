from dataclasses import dataclass, field
import polarimetry.manalysis.pathlib as pathlib
from matplotlib.colors import LogNorm
import pandas as pd
import numpy as np
from re import I
import matplotlib.pyplot as plt
import gzip
from typing import List, Dict, Iterator
from collections import Counter
from math import acos, degrees
from collections import Counter
import argparse

SI_Z = 0
CDTE_Z = -1.625
Z_TOL = 0.1  # cm, safe margin


@dataclass
class Interaction:
    index: int
    energy: float
    x: float
    y: float
    z: float


@dataclass
class TraEvent:
    event_id: int = None
    time: float = None
    event_type: str = None   # CO, PH, PA, etc
    sequence_length: int = 0
    interactions: List[Interaction] = field(default_factory=list)
    hits: List[Interaction] = field(default_factory=list)
    raw: Dict[str, List[str]] = field(default_factory=dict)


def count_event_types(filename: str):
    counts = Counter()

    for event in parse_tra(filename):
        #print(event)
        event_type = classify_event(event)
        counts[event_type] += 1

    return counts



def classify_event(event) -> str:
    detectors = set()

    for hit in event.hits:   # hits parsed from CH lines
        detectors.add(detector_from_z(hit.z))
    
    #print(detectors)
    if detectors == {"Si"}:
        return "Si-only"

    elif detectors == {"CdTe"}:
        return "CdTe-only"

    elif detectors == {"Si", "CdTe"}:
        return "Si+CdTe"   # THIS is your good Compton event

    else:
        return "Other"

def is_forward_compton(event) -> bool:
    if len(event.hits) < 2:
        return False

    first = event.hits[0]
    second = event.hits[1]

    return (
        detector_from_z(first.z) == "Si"
        and detector_from_z(second.z) == "CdTe"
    )


def parse_tra(filename: str) -> Iterator[TraEvent]:
    event = None
    
    with gzip.open(filename, "rt") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            parts = line.split()
            if not parts:
                continue
            tag = parts[0]
            
            # NEW EVENT on SE (yield previous, start new)
            if tag == "SE":
                if event is not None:
                    yield event
                event = TraEvent()
                continue  # Skip rest of SE line
                
            # ONLY process if we have active event
            if event is not None:
                if tag == "ET":
                    event.event_type = parts[1]
                elif tag == "ID":
                    event.event_id = int(parts[1])
                    id = int(parts[1])
                elif tag == "TI":
                    event.time = float(parts[1])
                elif tag == "SQ":
                    event.sequence_length = int(parts[1])
                elif tag == "PE":
                    energy = float(parts[1])
                elif tag == "PP":  # SINGLE PHOTON: PP x y z
                    hit = Interaction(
                        index=id,
                        x=float(parts[1]),
                        y=float(parts[2]),
                        z=float(parts[3]),
                        energy=energy  # PP has no energy, use PE above
                    )
                    event.hits.append(hit)
                elif tag == "CH":  # COMPTON: CH N x y z E ...
                    hit = Interaction(
                        index=int(parts[1]),
                        x=float(parts[2]),
                        y=float(parts[3]),
                        z=float(parts[4]),
                        energy=float(parts[5])
                    )
                    event.hits.append(hit)
                else:
                    event.raw.setdefault(tag, []).append(line)
                    
        # Yield final event
        if event is not None:
            yield event

def detector_from_z(z: float) -> str:
    if float(z) == 0:        
        return "Si"
    elif z == -1.625:
        return "CdTe"
    else:
        return "Other"

def check_event_detector(z, dist_between_det):
    # negative sign means cdte is always bellow the si
    if z > -(dist_between_det/2): # si 
        return 0
    else:
        return 1
def coordinate_transform(x, instrument_size):
    '''
    Apply coordinate transfrom from x,y = 0,0 in the center of the geometry to the left bottom corner of the detector
    '''
    return x + (instrument_size/2)

def get_pixel_id(x, pixel_size):
    '''
    Returns pix ID, for a given x,y coordinate. Need to compute coordinate_transform() first to be on the correct reference frame.
    '''
    ix = np.floor(x / pixel_size).astype(int) -1
    return ix

def get_matrix_id(x_pix, y_pix, matrix_size):
    '''
    matrix_size: number of pixels in x direction (assuming y direction has the same ammount of pix)
    '''
    pix_id = y_pix * matrix_size + x_pix
    return pix_id
    

def check_mega_event_insideDet(x, y, instrument_size):
    '''
    Sometimes megalib gives events that are ouside the detector. Check HomogeneousBeam661keV_config1.inc1.id1.tra.gz event ID 313, coordinat on CdTe is x=-5cm with reference on center of detector, its outside cdte detector
    x,y, on the detector ref frame
    '''
    if x > instrument_size or x < 0:
        return False
    elif y>instrument_size or y<0:
        return False
    else:
        return True


def print_parsing_tra_ascii():
    """Print PARSING TRA in ASCII art style"""
    art = """
██████╗  █████╗ ██████╗ ███████╗██╗███╗   ██╗ ██████╗     ████████╗██████╗  █████╗ 
██╔══██╗██╔══██╗██╔══██╗██╔════╝██║████╗  ██║██╔════╝     ╚══██╔══╝██╔══██╗██╔══██╗
██████╔╝███████║██████╔╝███████╗██║██╔██╗ ██║██║  ███╗       ██║   ██████╔╝███████║
██╔═══╝ ██╔══██║██╔══██╗╚════██║██║██║╚██╗██║██║   ██║       ██║   ██╔══██╗██╔══██║
██║     ██║  ██║██║  ██║███████║██║██║ ╚████║╚██████╔╝       ██║   ██║  ██║██║  ██║
╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚═╝╚═╝  ╚═══╝ ╚═════╝        ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝
    """
    print(art)

def print_final_table(filename, n_events, n_singles, n_double_compton, n_mult_compton, counts):
    """Print a compact table."""
    
    print("\n" + "=" * 52)
    print("EVENT STATISTICS")
    print("=" * 52)
    print(f"File: {filename}")
    print(f"Total: {n_events:,} events")
    print("-" * 52)
    
    # Create a simple table
    data = [
        ("Single Photon", n_singles, n_singles/n_events*100),
        ("Double Compton", n_double_compton, n_double_compton/n_events*100),
        ("Multi Compton", n_mult_compton, n_mult_compton/n_events*100),
        ("-" * 20, "-" * 10, "-" * 7),
        ("Si-only", counts['Si-only'], counts['Si-only']/n_events*100),
        ("CdTe-only", counts['CdTe-only'], counts['CdTe-only']/n_events*100),
        ("Si+CdTe", counts['Si+CdTe'], counts['Si+CdTe']/n_events*100),
        ("Other", counts['Other'], counts['Other']/n_events*100),
    ]
    
    for label, count, pct in data:
        if label.startswith("-"):
            print(f"{label:<20} {count:>10} {pct:>7}")
        else:
            print(f"{label:<20} {count:>10,} {pct:>6.2f}%")
    
    print("=" * 52)

if __name__ == '__main__':
    '''
    This parser takes .tra.gz files from revan megalib and identifies single and compton events (up to mult 7) in the data. It Parses the data to the PHEMTO 4x4 HED configuration and saves it into a .csv like file named *.t3pa.
    '''
    
    print_parsing_tra_ascii()

    parser = argparse.ArgumentParser(description='This parser takes .tra.gz files from revan megalib and identifies single and double compton events in the data. It Parses the data to the PHEMTO 4x4 HED configuration and saves it into a .csv like file named *.t3pa.')
    parser.add_argument('-f', '--filename', required=True, help='Full path of the .tra.gz file to parse')
    parser.add_argument('-o', '--output', required=True, help='Output path for the .t3pa files')
    parser.add_argument('-p', '--psf', default=0, help='Use this argument to limit events that happened on circular PSF. Usage: -p <psf radius in cm>, ie: -p 0.075 for 0.075 cm radius PSF')

    args = parser.parse_args()
    
    filename = args.filename
    output_folder = args.output # for the .t3pa
    pathlib.creat_dir(output_folder)


    ####### PARSING the .tra file #########
    tra_events = list(parse_tra(filename))
    n_events = len(tra_events)
    ######################################


    #Si: https://www.sciencedirect.com/science/article/pii/S0168900219310812
    #Si detector is the same for all configs
    dist = 1.5
    instrument_size_si = 6.656 #6.4 x 6.4 cm2, 4x4 MC2
    pix_size_si = 0.013 # cm = 0.13mm
    pix_matrix_size_si = 512 # 512x512

    # CdTe detector constants
    single_det_size = 1.6 # 1.6x1.6 cm2
    pix_size_cdte = 0.025 # cm = 0.25mm
    pix_per_det = 64 # 64x64 matrix
    instrument_size_cdte = single_det_size * 4# 1.6cm x cdte matrix_size 
    pix_matrix_size_cdte = pix_per_det * 4     # 64 pixels per det x cdte matrix_size
    

    event_lst, time_lst, energy_lst = [], [], []
    xpix_lst, ypix_lst, matrixID_lst = [], [], []
    x_lst, y_lst, z_lst, chip_id = [], [], [], []
    ouput_file_cntr = 0
    max_event_cnt = 1e4  # .t3pa number of events, polarimetry.py likes small file size for paralization
    df = pd.DataFrame()
    event_cnt = 0

    def process_single_hit(event, hit, dist, instr_size_si, pix_si, matrix_si, instr_size_cdte, pix_cdte, matrix_cdte, output_folder):
        '''
        Process one hit → pixel mapping on either cdte or si detectors
        '''
        global event_cnt, ouput_file_cntr, df, event_lst, time_lst, energy_lst
        global xpix_lst, ypix_lst, matrixID_lst, x_lst, y_lst, z_lst, chip_id
        # Detector selection
        detector = check_event_detector(hit.z, dist)
        if detector == 1:  # CdTe
            instrument_size = instr_size_cdte
            pix_size = pix_cdte
            pix_matrix_size = matrix_cdte
        else:  # Si
            instrument_size = instr_size_si
            pix_size = pix_si
            pix_matrix_size = matrix_si
        
        # Transform coordinates
        x_detRef = coordinate_transform(hit.x, instrument_size)
        y_detRef = coordinate_transform(hit.y, instrument_size)
        
        if not check_mega_event_insideDet(x_detRef, y_detRef, instrument_size):
            return  # Skip off-detector
        
        # Get pixel id for each deterctor, depends on detector characteristics, pix size, size etc...
        x_pix = get_pixel_id(x_detRef, pix_size)
        y_pix = get_pixel_id(y_detRef, pix_size)
        matrix_id = get_matrix_id(x_pix, y_pix, pix_matrix_size)
        
        event_lst.append(event.event_id)
        time_lst.append(event.time * 1e9)  # ns
        energy_lst.append(hit.energy)       # keV
        x_lst.append(hit.x)
        y_lst.append(hit.y)
        z_lst.append(hit.z)
        xpix_lst.append(x_pix)
        ypix_lst.append(y_pix)
        matrixID_lst.append(matrix_id)
        chip_id.append(detector)
        event_cnt += 1
        
        if event_cnt >= max_event_cnt:
            df = pd.DataFrame({
                            'Event': event_lst,
                            'Matrix Index': matrixID_lst,
                            'ToT (keV)': energy_lst,
                            'X': xpix_lst,
                            'Y': ypix_lst,
                            'Ns': time_lst,
                            'Overflow': chip_id  # 0=Si, 1=CdTe
                        })
            df.to_csv(f"{output_folder}/{ouput_file_cntr}.t3pa", index=False)
            # Reset for next chunk
            ouput_file_cntr += 1
            event_cnt = 0
            event_lst, time_lst, energy_lst = [], [], []
            xpix_lst, ypix_lst, matrixID_lst = [], [], []
            x_lst, y_lst, z_lst, chip_id = [], [], [], []

    
    ######## Transforming to detector Hits, Si and CdTe, saving in .t3pa #########
    n_singles = 0 
    n_double_compton = 0 
    n_mult_compton = 0

    PSF_radius = 0.075 # cm
    for event in tra_events:
        if event.event_type == "PH" and len(event.hits) == 1:
            # SINGLE PHOTON - existing logic (unchanged)
            hit = event.hits[0]
            
            if np.sqrt((hit.x)**2 + (hit.y)**2) > PSF_radius and args.psf != 0: # The reference frame of MEGAlib if center of Si and CdTe. Homogeneous Beam 1.5mm diameter centered at x,y=0,0
                continue # ignore events outside the PSF circle

            else:
                process_single_hit(event, hit, dist, instrument_size_si, pix_size_si, pix_matrix_size_si,
                              instrument_size_cdte, pix_size_cdte, pix_matrix_size_cdte, output_folder)
                n_singles += 1
        
        ## IMPLEMENT A SEXY VERSION OF THE FOLLOWING BULLSHIT...............
        elif event.event_type == "CO":
            # Check if sequence length matches number of hits
            if event.sequence_length == len(event.hits):
                if event.sequence_length == 2:
                    # DOUBLE COMPTON
                    # The idea is when PSF limit we save compton events that has at least one event inside PSF
                    hit1, hit2 = event.hits[0], event.hits[1]
                    r1 = np.sqrt((hit1.x**2) + (hit1.y**2))
                    r2 = np.sqrt((hit2.x**2) + (hit2.y**2))
                    if args.psf != 0:
                        if r1 <= PSF_radius or r2 <= PSF_radius: # If one of the comptons inside PSF are we save event
                            for hit in [hit1, hit2]:
                                process_single_hit(event, hit, dist, instrument_size_si, pix_size_si, pix_matrix_size_si,
                                              instrument_size_cdte, pix_size_cdte, pix_matrix_size_cdte, output_folder)
                            n_double_compton += 1
                        else:
                            continue #ignore event, the compton is fully outside PSF area
                    else: # in case we dont do PSF area cut
                        for hit in [hit1, hit2]:
                            process_single_hit(event, hit, dist, instrument_size_si, pix_size_si, pix_matrix_size_si,
                                          instrument_size_cdte, pix_size_cdte, pix_matrix_size_cdte, output_folder)
                        n_double_compton += 1
                    
                
                elif event.sequence_length == 3:
                    # TRIPLE COMPTON
                    hit1, hit2, hit3 = event.hits[0], event.hits[1], event.hits[2]
                    r1 = np.sqrt((hit1.x**2) + (hit1.y**2))
                    r2 = np.sqrt((hit2.x**2) + (hit2.y**2))
                    r3 = np.sqrt((hit3.x**2) + (hit3.y**2))
                    if args.psf != 0: # Same logic as in double event
                        if r1 <= PSF_radius or r2 <= PSF_radius or r3 <= PSF_radius:
                            for hit in [hit1, hit2, hit3]:
                                process_single_hit(event, hit, dist, instrument_size_si, pix_size_si, pix_matrix_size_si,
                                                  instrument_size_cdte, pix_size_cdte, pix_matrix_size_cdte, output_folder)
                            n_mult_compton += 1
                        else:
                            continue
                    else:
                        for hit in [hit1, hit2, hit3]:
                            process_single_hit(event, hit, dist, instrument_size_si, pix_size_si, pix_matrix_size_si,
                                              instrument_size_cdte, pix_size_cdte, pix_matrix_size_cdte, output_folder)
                        n_mult_compton += 1
                
                elif event.sequence_length == 4:
                    # QUADRUPLE COMPTON
                    hit1, hit2, hit3, hit4 = event.hits[0], event.hits[1], event.hits[2], event.hits[3]
                    r1 = np.sqrt((hit1.x**2) + (hit1.y**2))
                    r2 = np.sqrt((hit2.x**2) + (hit2.y**2))
                    r3 = np.sqrt((hit3.x**2) + (hit3.y**2))
                    r4 = np.sqrt((hit4.x**2) + (hit4.y**2))
                    if args.psf != 0: # Same logic as in double event
                        if r1 <= PSF_radius or r2 <= PSF_radius or r3 <= PSF_radius or r4 <= PSF_radius:
                            for hit in [hit1, hit2, hit3, hit4]:
                                process_single_hit(event, hit, dist, instrument_size_si, pix_size_si, pix_matrix_size_si,
                                                  instrument_size_cdte, pix_size_cdte, pix_matrix_size_cdte, output_folder)
                            n_mult_compton =+ 1
                        else:
                            continue
                    else:
                        for hit in [hit1, hit2, hit3, hit4]:
                            process_single_hit(event, hit, dist, instrument_size_si, pix_size_si, pix_matrix_size_si,
                                              instrument_size_cdte, pix_size_cdte, pix_matrix_size_cdte, output_folder)
                        n_mult_compton =+ 1

                
                elif event.sequence_length == 5:
                    # QUINTUPLE COMPTON
                    hit1, hit2, hit3, hit4, hit5 = event.hits[0], event.hits[1], event.hits[2], event.hits[3], event.hits[4]
                    r1 = np.sqrt((hit1.x**2) + (hit1.y**2))
                    r2 = np.sqrt((hit2.x**2) + (hit2.y**2))
                    r3 = np.sqrt((hit3.x**2) + (hit3.y**2))
                    r4 = np.sqrt((hit4.x**2) + (hit4.y**2))
                    r5 = np.sqrt((hit5.x**2) + (hit5.y**2))
                    if args.psf != 0: # Same logic as in double event
                        if r1 <= PSF_radius or r2 <= PSF_radius or r3 <= PSF_radius or r4 <= PSF_radius or r5 <= PSF_radius:
                            for hit in [hit1, hit2, hit3, hit4, hit5]:
                                process_single_hit(event, hit, dist, instrument_size_si, pix_size_si, pix_matrix_size_si,
                                                  instrument_size_cdte, pix_size_cdte, pix_matrix_size_cdte, output_folder)
                            n_mult_compton += 1
                        else:
                            continue
                    else:
                        for hit in [hit1, hit2, hit3, hit4, hit5]:
                            process_single_hit(event, hit, dist, instrument_size_si, pix_size_si, pix_matrix_size_si,
                                              instrument_size_cdte, pix_size_cdte, pix_matrix_size_cdte, output_folder)
                        n_mult_compton += 1
                
                elif event.sequence_length == 6:
                    # SEXTUPLE COMPTON
                    hit1, hit2, hit3, hit4, hit5, hit6 = event.hits[0], event.hits[1], event.hits[2], event.hits[3], event.hits[4], event.hits[5]
                    r1 = np.sqrt((hit1.x**2) + (hit1.y**2))
                    r2 = np.sqrt((hit2.x**2) + (hit2.y**2))
                    r3 = np.sqrt((hit3.x**2) + (hit3.y**2))
                    r4 = np.sqrt((hit4.x**2) + (hit4.y**2))
                    r5 = np.sqrt((hit5.x**2) + (hit5.y**2))
                    r6 = np.sqrt((hit6.x**2) + (hit6.y**2))
                    if args.psf != 0: # Same logic as in double event
                        if r1 <= PSF_radius or r2 <= PSF_radius or r3 <= PSF_radius or r4 <= PSF_radius or r5 <= PSF_radius or r6 <= PSF_radius:
                            for hit in [hit1, hit2, hit3, hit4, hit5, hit6]:
                                process_single_hit(event, hit, dist, instrument_size_si, pix_size_si, pix_matrix_size_si,
                                                  instrument_size_cdte, pix_size_cdte, pix_matrix_size_cdte, output_folder)
                            n_mult_compton += 1
                        else:
                            continue
                    else:
                        for hit in [hit1, hit2, hit3, hit4, hit5, hit6]:
                            process_single_hit(event, hit, dist, instrument_size_si, pix_size_si, pix_matrix_size_si,
                                              instrument_size_cdte, pix_size_cdte, pix_matrix_size_cdte, output_folder)
                        n_mult_compton += 1
                
                elif event.sequence_length == 7:
                    # SEPTUPLE COMPTON
                    hit1, hit2, hit3, hit4, hit5, hit6, hit7 = event.hits[0], event.hits[1], event.hits[2], event.hits[3], event.hits[4], event.hits[5], event.hits[6]
                    r1 = np.sqrt((hit1.x**2) + (hit1.y**2))
                    r2 = np.sqrt((hit2.x**2) + (hit2.y**2))
                    r3 = np.sqrt((hit3.x**2) + (hit3.y**2))
                    r4 = np.sqrt((hit4.x**2) + (hit4.y**2))
                    r5 = np.sqrt((hit5.x**2) + (hit5.y**2))
                    r6 = np.sqrt((hit6.x**2) + (hit6.y**2))
                    r7 = np.sqrt((hit7.x**2) + (hit7.y**2))
                    if args.psf != 0: # Same logic as in double event
                        if r1 <= PSF_radius or r2 <= PSF_radius or r3 <= PSF_radius or r4 <= PSF_radius or r5 <= PSF_radius or r6 <= PSF_radius or r7 <= PSF_radius:
                            for hit in [hit1, hit2, hit3, hit4, hit5, hit6, hit7]:
                                process_single_hit(event, hit, dist, instrument_size_si, pix_size_si, pix_matrix_size_si,
                                                  instrument_size_cdte, pix_size_cdte, pix_matrix_size_cdte, output_folder)
                            n_mult_compton += 1
                        else:
                            continue
                    else:
                        for hit in [hit1, hit2, hit3, hit4, hit5, hit6, hit7]:
                            process_single_hit(event, hit, dist, instrument_size_si, pix_size_si, pix_matrix_size_si,
                                              instrument_size_cdte, pix_size_cdte, pix_matrix_size_cdte, output_folder)
                        n_mult_compton += 1



            #hit1, hit2 = event.hits[0], event.hits[1]  # First=Si scatter, Second=CdTe absorb
            #
            #process_single_hit(event, hit1, dist, instrument_size_si, pix_size_si, pix_matrix_size_si,
            #                  instrument_size_cdte, pix_size_cdte, pix_matrix_size_cdte, output_folder)
            #process_single_hit(event, hit2, dist, instrument_size_si, pix_size_si, pix_matrix_size_si,
            #                  instrument_size_cdte, pix_size_cdte, pix_matrix_size_cdte, output_folder)


    df = pd.DataFrame({
                    'Event': event_lst,
                    'Matrix Index': matrixID_lst,
                    'ToT (keV)': energy_lst,
                    'X': xpix_lst,
                    'Y': ypix_lst,
                    'Ns': time_lst,
                    'Overflow': chip_id  # 0=Si, 1=CdTe
                })
    df.to_csv(f"{output_folder}/{ouput_file_cntr}.t3pa", index=False)

    ##### Debug Plots #####
    df_cdte = df[df['Overflow'] == 1] ## CdTe
    pixel_energy = df_cdte.groupby(['X', 'Y'])['ToT (keV)'].sum().reset_index()
    energy_map = np.zeros((pix_matrix_size_cdte, pix_matrix_size_cdte))
    for _, row in pixel_energy.iterrows():
        x = int(row['X'])
        y = int(row['Y'])
        energy_map[y, x] = row['ToT (keV)']  # Fill with summed energy

    plt.figure(figsize=(16, 12))
    plt.subplot(2,2,1)
    plt.imshow(energy_map, cmap='jet', origin='lower', interpolation='nearest')
    plt.colorbar(label='Total energy deposited (keV)')
    plt.xlabel('X pixel')
    plt.ylabel('Y pixel')
    plt.title('CdTe: Energy Summed per Pixel (GroupBy)')

    plt.subplot(2,2,3)
    try:
        im = plt.imshow(energy_map, cmap='jet', origin='lower', 
                        norm=LogNorm(vmin=energy_map[energy_map > 0].min(), 
                                    vmax=energy_map.max()),
                        interpolation='nearest')
    except:
        im = plt.imshow(energy_map, cmap='jet', origin='lower', 
                    interpolation='nearest')
    plt.colorbar(im, label='Total energy deposited (keV)')
    plt.xlabel('X pixel')
    plt.ylabel('Y pixel')
    plt.title('CdTe: Energy Summed per Pixel (Log Scale)')

    df_si = df[df['Overflow'] == 0] # Si detector
    pixel_energy_si = df_si.groupby(['X', 'Y'])['ToT (keV)'].sum().reset_index()
    energy_map_si = np.zeros((pix_matrix_size_si, pix_matrix_size_si))
    for _, row in pixel_energy_si.iterrows():
        x = int(row['X'])
        y = int(row['Y'])
        energy_map_si[y, x] = row['ToT (keV)']

    plt.subplot(2,2,2)
    plt.imshow(energy_map_si, cmap='jet', origin='lower', interpolation='nearest')
    plt.colorbar(label='Total energy (keV)')
    plt.title('Si: Linear Scale')
    plt.xlabel('X pixel')
    plt.ylabel('Y pixel')

    plt.subplot(2,2,4)
    im = plt.imshow(energy_map_si, cmap='jet', origin='lower',
                    norm=LogNorm(vmin=energy_map_si[energy_map_si > 0].min(),
                                vmax=energy_map_si.max()),
                    interpolation='nearest')
    plt.colorbar(im, label='Total energy (keV)')
    plt.title('Si: Log Scale')
    plt.xlabel('X pixel')
    plt.ylabel('Y pixel')

    plt.tight_layout()
    plt.savefig(f"{output_folder}/0_imgDebug.png")
    plt.close()


    ## efficiencies
    si_only = []; cdte_only = []; si_cdte = []; total_events = []
    e = []
    n = []
    n_compton = 0
    n_2site = 0
    
    # This also parses the events
    counts = count_event_types(filename)
    
    si_only.append(counts['Si-only'])
    cdte_only.append(counts['CdTe-only']) 
    si_cdte.append(counts['Si+CdTe'])
    total_events.append(counts['Si-only'] + counts['CdTe-only'] + counts['Si+CdTe'])
    print_final_table(filename, n_events, n_singles, n_double_compton, n_mult_compton, counts)

    with open(f"{output_folder}/parse_event_counter.txt", 'w') as f:
        f.write(f"# Total events : {n_events}\n# Total singles : {n_singles}\n# Total double Compton : {n_double_compton}\n# Total mult Compton : {n_mult_compton}")
    #si_eff = np.array(si_only) / N_generated
    #cdte_eff = np.array(cdte_only) / N_generated  
    #si_cdte_eff = np.array(si_cdte) / N_generated
    #total_eff = np.array(total_events) / N_generated

    #fig, ax = plt.subplots(figsize=(8, 6))

    #ax.plot(log_E, si_eff*100, 'o-', color='skyblue', linewidth=2, markersize=8, 
    #        label='Si-only', alpha=0.8)
    #ax.plot(log_E, cdte_eff*100, 's-', color='salmon', linewidth=2, markersize=8, 
    #        label='CdTe-only', alpha=0.8)
    #ax.plot(log_E, si_cdte_eff*100, 'D-', color='darkgreen', linewidth=2, markersize=8, 
    #        label='Si+CdTe (Compton)', alpha=0.9)
    #ax.plot(log_E, total_eff*100, 'D-', color='dimgray', linewidth=3, markersize=10, 
    #        label='Total', alpha=0.9)

    #ax.set_xlabel('Energy (keV)', fontsize=14, fontweight='bold')
    #ax.set_ylabel('Efficiency (%)', fontsize=14, fontweight='bold')
    #ax.grid(True, alpha=0.3)
    #ax.legend(fontsize=12, loc='upper right', framealpha=0.95)

    #plt.title('Phemto 4x4 Efficiency vs Energy\n(1×10⁶ MC events per energy bin)', 
    #          fontsize=15, fontweight='bold', pad=20)
    #plt.yscale('log')  # Log scale for efficiency drop-off
    #plt.tight_layout()
    #plt.show()

