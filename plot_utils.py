import pandas as pd
import matplotlib.pyplot as plt   

def plot_eccentricity(csv_file1: str,
                      csv_file2: str,
                      label1: str = '1B',
                      label2: str = '4B',
                      save_file: str = None) -> None:
    # set figure size
    plt.figure(figsize=(8, 5))
    plt.rcParams.update({'font.size': 18})

    csv1 = pd.read_csv(csv_file1)
    csv2 = pd.read_csv(csv_file2)

    csv1 = csv1.iloc[:901]
    csv2 = csv2.iloc[:901]
    
    # csv1 blue dotted line
    # csv2 red solid line
    plt.plot(csv1['score'], label=label1, linestyle='dotted', color='blue')
    plt.plot(csv2['score'], label=label2, linestyle='solid', color='red')

    # draw gray line at 0.5
    plt.axhline(y=0.5, color='gray', linestyle='--')
    plt.legend()

    # y-axis : 0, 0.5, 1.0, x-axis : 0, 0.3, 0.6, 0.9
    plt.xlim(0, 0.9)
    plt.ylim(0, 1.0)
    plt.xticks([0, 0.3, 0.6, 0.9])
    plt.yticks([0, 0.5, 1.0])

    plt.xlabel('Eccentricity')
    plt.ylabel('Score')
    
    
    plt.tight_layout()
    plt.savefig(save_file)

def plot_polygon(csv_file1: str,
                 csv_file2: str,
                 label1: str = '1B',
                 label2: str = '4B',
                 save_file: str = None) -> None:
    # set figure size
    plt.figure(figsize=(8, 5))
    plt.rcParams.update({'font.size': 18})

    csv1 = pd.read_csv(csv_file1)
    csv2 = pd.read_csv(csv_file2)

    csv1 = csv1.iloc[:28]
    csv2 = csv2.iloc[:28]
    
    plt.plot(csv1['score'], label=label1, linestyle='dotted', color='blue')
    plt.plot(csv2['score'], label=label2, linestyle='solid', color='red')

    plt.axhline(y=0.5, color='gray', linestyle='--')
    plt.legend()

    plt.xlim(4, 30.0)
    plt.ylim(0, 1.0)
    plt.xticks([6, 12, 18, 24, 30])
    plt.yticks([0, 0.5, 1.0])

    plt.xlabel('Polygon')
    plt.ylabel('Score')
    
    plt.tight_layout()
    plt.savefig(save_file)

def plot_size(csv_file1: str,
              csv_file2: str,
              label1: str = '1B',
              label2: str = '4B',
              save_file: str = None) -> None:
    # set figure size
    plt.figure(figsize=(8, 5))
    plt.rcParams.update({'font.size': 18})

    csv1 = pd.read_csv(csv_file1)
    csv2 = pd.read_csv(csv_file2)
    
    plt.plot(csv1['score'], label=label1, linestyle='dotted', color='blue')
    plt.plot(csv2['score'], label=label2, linestyle='solid', color='red')

    plt.axhline(y=0.5, color='gray', linestyle='--')
    plt.legend()

    plt.xlim(0, 199.0)
    plt.ylim(0, 1.0)
    plt.xticks([0, 30, 60, 90, 120, 150, 180])
    plt.yticks([0, 0.5, 1.0])

    plt.xlabel('Size')
    plt.ylabel('Score')
    
    plt.tight_layout()
    plt.savefig(save_file)