import argparse
from plot_utils import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file1", type=str, default="outputs/sensitivity/InternVL2_5-1B_eccentricity.csv")
    parser.add_argument("--csv_file2", type=str, default="outputs/sensitivity/InternVL2_5-4B_eccentricity.csv")
    parser.add_argument("--label1", type=str, default="1B")
    parser.add_argument("--label2", type=str, default="4B")
    parser.add_argument("--save_file", type=str, default="outputs/sensitivity/InternVL2_5-1B_4B_eccentricity.pdf")
    args = parser.parse_args()

    if "eccentricity" in args.save_file:
        plot_eccentricity(args.csv_file1,
                          args.csv_file2,
                          args.label1, 
                          args.label2,
                          args.save_file)
    elif "poly" in args.save_file:
        plot_polygon(args.csv_file1,
                     args.csv_file2,
                     args.label1, 
                     args.label2,
                     args.save_file)
    elif "size" in args.save_file:
        plot_size(args.csv_file1,
                  args.csv_file2,
                  args.label1, 
                  args.label2,
                  args.save_file)

