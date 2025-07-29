


#  Unchanged imports from original file
import pickle     
import struct    
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path  #



# Unchanged function from original file
def calc_linearity(Codes16):
    Codes14 = Codes16 // 4
    sortCodes14 = np.sort(Codes14)
    minbin = sortCodes14[30]
    maxbin = sortCodes14[-30]
    yoffset = ((sortCodes14[1] + sortCodes14[-2]) // 2) - 8192
    minCodes16 = np.amin(Codes16)
    maxCodes16 = np.amax(Codes16)
    minCodes14 = np.amin(Codes14)
    maxCodes14 = np.amax(Codes14)
    print("Min/max code, spread (16bit)=", minCodes16, maxCodes16, maxCodes16 - minCodes16)
    print("Min/max code, spread (14bit)=", minCodes14, maxCodes14, maxCodes14 - minCodes14)
    print("Second Min/max code, offset (14bit)=", sortCodes14[1], sortCodes14[-2], yoffset)
    del sortCodes14

    bins = np.append(np.insert(np.arange(minbin, maxbin + 2) - 0.5, 0, 0.0), 16384.5)
    h, binedges = np.histogram(Codes14, bins)
    midADCmean = np.mean(h[7500:8200])
    midADCstd = np.std(h[7500:8200])
    print('midADCmean: ', midADCmean)
    print('midADCstd: ', midADCstd)
    ch = np.cumsum(h)
    histosum = np.sum(h)
    end = np.size(ch)
    T = -np.cos(np.pi * ch / histosum)
    hlin = np.subtract(T[1:end], T[0:end - 1])
    TRUNC = 30
    hlin_size = np.size(hlin)
    hlin_trunc = hlin[TRUNC:hlin_size - TRUNC]
    lsb = np.average(hlin_trunc)
    dnl = np.insert(hlin_trunc / lsb - 1, 0, 0.0)
    inl = np.cumsum(dnl)
    inlNorm = inl - np.mean(inl)
    code = np.linspace(minbin + TRUNC, maxbin - TRUNC, np.size(dnl)).astype(np.uint16)
    print(f"Range of code: {minbin + TRUNC} - {maxbin - TRUNC}, difference: {maxbin - TRUNC - (minbin + TRUNC)}\nSize of 'code': {np.size(code)}")
    return code, dnl, inlNorm, midADCmean, midADCstd


# ➕ NEW FUNCTION replacing the original `calc_plot_dnl_inl()` 

#  version does NOT require deserialize or a raw FIFO input
def plot_dnl_from_pickle(bin_path: str, save_dir: str = ".", plot_subset: int = 8):
    # Load your .bin file pickld python dict.
    with open(bin_path, 'rb') as f:
        data = pickle.load(f)

    all_channel_data = []

    # loop through all histdata-containing entries
    for onekey in data:
        if "histdata" not in onekey:
            continue

        cfgdata = data[onekey]
        histdata = cfgdata[1]  # second element = list of channel data blobs

        # extract 16-bit ADC words from each channel
        for ch_data in histdata:
            if not ch_data:
                continue
            num_16bwords = len(ch_data) // 2
            words16b = struct.unpack_from(f"<{num_16bwords}H", ch_data)
            all_channel_data.append(np.array(words16b, dtype=np.uint16))

    # apply calc_linearity() to each channel
    results = [calc_linearity(codes) for codes in all_channel_data]

    # store in df
    df = pd.DataFrame(results, columns=['code', 'dnl', 'inlNorm', 'midADCmean', 'midADCstd'])
    df.index.name = 'Channel'

    # plot DNL INL for first N channels
    for ch in range(min(len(df), plot_subset)):
        fig, axs = plt.subplots(2, 1, figsize=(10, 6))

        axs[0].plot(df.at[ch, 'code'], df.at[ch, 'dnl'])
        axs[0].set_title(f'DNL - Channel {ch}')
        axs[0].set_ylabel('DNL [LSB]')
        axs[0].grid(True)

        axs[1].plot(df.at[ch, 'code'], df.at[ch, 'inlNorm'])
        axs[1].set_title(f'INL - Channel {ch}')
        axs[1].set_xlabel('ADC Code')
        axs[1].set_ylabel('INL [LSB]')
        axs[1].grid(True)

        fig.tight_layout()
        fig.savefig(Path(save_dir) / f"channel_{ch}_dnl_inl.png")
        plt.close(fig)

    return df


# to test your new bin files directly
if __name__ == "__main__":
    path_to_bin = "your_file_path_here.bin"  #  file path
    save_dir = "./plots"                     # output dir if you want
    plot_dnl_from_pickle(path_to_bin, save_dir, plot_subset=8)
