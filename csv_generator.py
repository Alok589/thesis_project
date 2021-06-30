import os, glob
import pandas as pd

# path = "/home/thesis_2/Res_Unet_csvfiles"

# all_files = glob.glob(os.path.join(path, "*.csv"))

# all_df = []
# for f in all_files:
#     df = pd.read_csv(f, sep=",")
#     df["file"] = f.split("/")[-1]
#     all_df.append(df)

# merged_df = pd.concat(all_df, ignore_index=True, sort=True)
# pd.DataFrame(merged_df).to_csv("/home/thesis_2/Res_Unet_csvfiles/Res_Unet.csv")


df = pd.read_csv("/home/thesis_2/Res_Unet_csvfiles/Res_Unet.csv")
mean_psnr = df["psnr"].mean()
mean_ssim = df["ssim"].mean()
max_psnr = df["psnr"].max()
min_ssim = df["ssim"].min()
print("mean_psnr", mean_psnr)
print("max_psnr", max_psnr)
print("mean_ssim", mean_ssim)
print("min_ssim", min_ssim)

