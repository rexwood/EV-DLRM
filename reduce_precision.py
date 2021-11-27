#!/usr/bin/env python

import sys
import argparse
import os
import pandas as pd 
import numpy as np
from os import listdir
from pathlib import Path
from array import *
from tqdm import tqdm

# save to a file
def write_to_file(df, filePath):
    base_dir = Path(filePath).parent
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    df.to_csv(filePath, index=False, header=True, sep=',')
    print("===== output file : " + filePath)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-file", help="File path of the raw ev data",type=str)
    # parser.add_argument("-out_dir", help="File path of the output directory",type=str)
    parser.add_argument("-read_as", help="the ev data can be read as fp32 or fp16",type=str)
    parser.add_argument("-new_precision", help="the new precision, should be lower than read_as",type=str)
    args = parser.parse_args()
    if (not args.file) or (not args.new_precision) or (not args.read_as):
        print("ERROR: You must provide these 2 arguments: -file <the input file> -read_as <read as fp32 or fp16> -new_precision <precision lower than read_as>")
        exit(-1)
    else:
        # Create output directory at the PARENT dir
        inFile = args.file
        parentDir = str(Path(inFile).parent)
        fileName = os.path.basename(inFile) 
        outFile = str(Path(parentDir).parent) + "/ev-table-" + args.new_precision + "/" + fileName
        
        print("===== Read ev data as " + args.read_as)

        if (args.read_as == "fp32"):
            df = pd.read_csv(inFile, dtype=object, delimiter=',').astype(np.float32)
        elif (args.read_as == "fp16"):
            df = pd.read_csv(inFile, dtype=object, delimiter=',').astype(np.float16)
        else:
            print("ERROR: Can't understand the read_as format : " + args.read_as)
            exit(-1)

        print("prev = " + str(df['0'].iloc[0]) + " " + str(df['0'].iloc[1]) + " " + str(df['0'].iloc[2]))

        # reduce the precision
        if (args.new_precision == "32"):
            print("   Reduce the precision to fp32")
            df = pd.read_csv(inFile, dtype=object, delimiter=',').astype(np.float32)
        elif (args.new_precision == "16"):
            print("   Reduce the precision to fp16")
            df = pd.read_csv(inFile, dtype=object, delimiter=',').astype(np.float16)
        elif (args.new_precision == "0"):
            print("   Reduce the precision to 0 (FOR TESTING)")
            cols = df.columns
            for col in cols:
                # Make all value to 0
                df[col] = df[col].apply(lambda x: 0)
        elif (args.new_precision == "8"):
            print("   Reduce the precision to 8bit")
            min_val = df['0'].min()
            max_val = df['0'].max()
            cols = df.columns
            for col in cols:
                if(df[col].max() > max_val):
                    max_val = df[col].max()

                if(df[col].min() < min_val):
                    min_val = df[col].min()
            # the min and max out of all rows and columns
            print("      Min = " + str(min_val))
            print("      Max = " + str(max_val))

            # normalizing the range
            for col in cols:
                # range is -0.127 ... 0.127
                df[col] = df[col].apply(lambda x: round(((x - min_val)/(max_val - min_val) -0.5 ) * 2 * 0.127, 3))

        elif (args.new_precision == "4"):
            print("   Reduce the precision to 4bit")
            min_val = df['0'].min()
            max_val = df['0'].max()
            cols = df.columns
            for col in cols:
                if(df[col].max() > max_val):
                    max_val = df[col].max()

                if(df[col].min() < min_val):
                    min_val = df[col].min()
            # the min and max out of all rows and columns
            print("      Min = " + str(min_val))
            print("      Max = " + str(max_val))

            # normalizing the range
            for col in cols:
                # range is -0.7 ... 0.7
                df[col] = df[col].apply(lambda x: round(((x - min_val)/(max_val - min_val) -0.5 ) * 2 * 0.7, 1))
        else:
            print("ERROR: Can't understand the new_precision format : " + args.new_precision)
            exit(-1)

        print("now  = " + str(df['0'].iloc[0]) + " " + str(df['0'].iloc[1]) + " " + str(df['0'].iloc[2]))

        write_to_file(df, outFile)





