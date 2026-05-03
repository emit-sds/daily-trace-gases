### Fully Automatic Trace Gas Plume Detection
### Author: Vit Ruzicka, 2026

import os.path
from timeit import default_timer as timer
import argparse
from pipeline.run_for_ch4 import run_for_ch4
from pipeline.run_for_trace_gas import run_for_trace_gas
from utils.rio_utils import mkdir

parser = argparse.ArgumentParser(description='Codebase: Fully Automatic Trace Gas Plume Detection.')
parser.add_argument('-gas', help='Which trace gas? (options: ch4, nh3, no2 and co)', default='ch4')
parser.add_argument('-tile', help='Which EMIT tile to run? (e.g. EMIT_L1B_RAD_001_20260102T143123_2600209_005)', default='EMIT_L1B_RAD_001_20260102T143123_2600209_005')
parser.add_argument('-raws_folder', help='Folder to store intermediate files', default='./run_data/intermediates_folder')
parser.add_argument('-results_folder', help='Folder to save results to', default='./run_data/results')

def detect_trace_gas(gas, tile_ID, results_folder, raws_folder):
    print("Running detection of", gas, "in EMIT tile:", tile_ID)
    raws_folder = os.path.join(raws_folder, tile_ID)
    results_folder = os.path.join(results_folder, tile_ID)
    mkdir(raws_folder)
    mkdir(results_folder)

    if gas == "ch4":
        # Use the methane pathway:
        run_for_ch4(tile_ID, results_folder, raws_folder)

    elif gas in ["nh3", "no2", "co"]:
        # Use the other trace gases pathway:
        run_for_trace_gas(gas, tile_ID, results_folder, raws_folder)

    else:
        print("Trace gas", gas, "not supported!")
        assert False

if __name__ == '__main__':
    args = parser.parse_args()
    start = timer()

    tile_ID = args.tile
    gas = args.gas

    results_folder = args.results_folder
    raws_folder = args.raws_folder

    # ### DEMO LOCAL OVERRIDES
    # r = "."
    # tile_ID = "EMIT_L1B_RAD_001_20260102T143123_2600209_005"
    # gas = "ch4"
    # tile_ID = "EMIT_L1B_RAD_001_20250418T090845_2510806_029"
    # gas = "no2"
    # tile_ID = "EMIT_L1B_RAD_001_20231223T123029_2335708_008"
    # gas = "nh3"
    # tile_ID = "EMIT_L1B_RAD_001_20230623T031429_2317402_015"
    # gas = "co"
    #
    # raws_folder = r+"/run_data/intermediates_folder"
    # results_folder = r+"/run_data/results"
    # ###

    detect_trace_gas(gas, tile_ID, results_folder, raws_folder)

    end = timer()
    time = (end - start)
    print("This run took "+str(time)+"s ("+str(time/60.0)+"min)")