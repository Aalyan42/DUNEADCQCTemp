
print(f"\n--- BIN DEBUG ---")
print(f"minbin: {minbin}, maxbin: {maxbin}")
print(f"First 10 bins: {bins[:10]}")
print(f"Last 10 bins: {bins[-10:]}")
print(f"bins increasing? {np.all(np.diff(bins) > 0)}")
print(f"Any NaNs? {np.isnan(bins).any()}")
print(f"bins dtype: {bins.dtype}")

