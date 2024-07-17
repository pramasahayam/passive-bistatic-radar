import numpy as np
from pyapril.detector import cc_detector_ons
from pyapril import clutterCancellation as CC
from pyapril.caCfar import CA_CFAR
from pyapril.RDTools import export_rd_matrix_img

# Import libraries to make gif from plot images
from glob import glob
import contextlib
from PIL import Image
from os import remove
import time

import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from joblib import Parallel, delayed

start_time = time.time()

# Paramaters
# Follow working directory format, with "pyApril test.ipynb" + "Data", "Images", and "GIFs" folders in the working folder
# Remember to move old gifs out of ./GIFS/ if the name is same as new one, otherwise it will be overwritten

# Signal parameters - should match expected parameters of recorded data
sample_rate = 3 * 10**6  # sample rate
max_bistatic_range = 2**7  # should be power of 2 to work with cc_detector_ons
max_doppler = 250  # should be divisible by 10 to work with export_rd_matrix_img
N = 2**21  # samples to use in cross correlation, should be power of 2, 19-22 work well at 3-6 Msamples/s

# Iteration parameters
interval = int(5 * 10**5)  # interval length in number of samples, 5 or 10 *10**5 seem to work well for gif output
dyn_range = 10 * np.log10(N)  # don't touch, used for plotting function's colormap
td_filter_dimension = max_bistatic_range  # don't touch, used for CC.Wiener_SMI_MRE

# CFAR parameters - adjust to produce best result from data
# tried various CFAR parameters, fairly selective and seems to produce decent results
win_param = [10, 10, 5, 5]
threshold = 12

gif_name = ("s" + (str(int(np.log2(N)))) + "_r" + str(max_bistatic_range) + "_d" + str(max_doppler))

ref_ch = np.fromfile("./Data/rx1.bin", np.complex64)
surv_ch = np.fromfile("./Data/rx2.bin", np.complex64, count=ref_ch.size)

start_indices = []
offset = 0
while ref_ch.size >= (offset + N):
    start_indices.append(offset)
    offset += interval


def process_sample(start_pos):
    # multiprocessing implementation, also used for joblib implementation
    ref_ch_sample = ref_ch[start_pos : start_pos + N]
    surv_ch_sample = surv_ch[start_pos : start_pos + N]

    # multithreading implementation - doesn't work
    # ref_ch_sample = np.fromfile("./Data/rx1.bin",np.complex64,offset=start_pos,count=N)
    # surv_ch_sample = np.fromfile("./Data/rx2.bin",np.complex64,offset=start_pos,count=N)

    rd_matrix = cc_detector_ons(
        ref_ch_sample,
        surv_ch_sample,
        sample_rate,
        max_doppler,
        max_bistatic_range,
    )

    # if start_indicies.index(start_pos) == 0: # comment/uncomment for CFAR
    #     CFAR = CA_CFAR(win_param, threshold, rd_size=rd_matrix.shape)
    #     gif_name += "_cfar" # adding cfar parameter to end of gif name
    # rd_matrix = CFAR(rd_matrix)

    export_rd_matrix_img(
        fname="./Images/" + str(start_indices.index(start_pos)) + ".png",
        rd_matrix=rd_matrix,
        max_Doppler=max_doppler,
        ref_point_range_index=0,
        ref_point_Doppler_index=0,
        box_color="None",
        box_size=0,
        dyn_range=dyn_range,
        dpi=200,
        interpolation="sinc",
        cmap="jet",
    )
    print(f"Frame {start_indices.index(start_pos)}: Image Saved")


def savegif():
    print("Saving GIF...\n")

    with contextlib.ExitStack() as stack:
        imgs = (stack.enter_context(Image.open(f)) for f in sorted(glob("./Images/*.png"), key=lambda fname: int(fname[9:-4])))
        img = next(imgs)
        img.save(
            fp="./GIFs/" + gif_name + ".gif",
            format="GIF",
            append_images=imgs,
            save_all=True,
            duration=100,
            loop=0,
        )

    print(f"GIF Saved as {gif_name + '.gif'}")


if __name__ == "__main__":
    print(f"\nTotal Data Length: {ref_ch.size/sample_rate} (s)\nNumber of frames to process: {len(start_indices)}")
    print(f"\nParameters:\nMax Bistatic Range: {max_bistatic_range} (km)\nMax Doppler Frequency: {max_doppler} (Hz)")
    print(f"Sample Length: {N/sample_rate:.4} (s)\nIteration interval: {interval/sample_rate:.4} (s)")

    files = glob("./Images/*.png")
    for f in files:
        remove(f)

    # print("Performing clutter cancellation...")
    # try: # save filtered surv_ch to file to allow future runs on dataset to save time on clutter cancellation
    #     surv_ch = np.fromfile("./Data/rx2_cc.bin", np.complex64, count=ref_ch.size)
    #     print("surv_ch opened from rx2_cc.bin")
    # except:
    #     print("No rx2_cc.bin file was found, filtering surv_ch")
    #     surv_ch, w = CC.Wiener_SMI_MRE(ref_ch, surv_ch, td_filter_dimension)
    #     np.save("./Data/rx2_cc.bin",surv_ch)
    #     print("Filtered surv_ch saved to rx2_cc.bin")
    # gif_name += "_cc"
    # print("Clutter cancellation complete")

    # multiprocessing implementation - somewhat slower than joblib, adjust # of processes accordingly
    # print("\nStarting multiprocessing")
    # pool = multiprocessing.Pool(processes=6)
    # pool.map(process_sample, start_indices)
    # pool.close()
    # pool.join()

    # multithreading implementation - doesn't work
    # print("\nStarting multithreading")
    # ref_ch = []
    # surv_ch = []
    # with ThreadPoolExecutor(max_workers=8) as executor:
    #     executor.map(process_sample, start_indices)

    # joblib.Parallel implementation - best option, adjust n_jobs accordingly (-2 is all but one processor)
    print("\nStarting joblib.Parallel processing")
    Parallel(n_jobs=-2)(delayed(process_sample)(start_pos) for start_pos in start_indices)

    savegif()

    end_time = time.time()
    runtime = end_time - start_time
    print(f"Script runtime: {runtime:.4f} (s)")
