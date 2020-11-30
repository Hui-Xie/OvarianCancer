# read the error array of grid-searching lambda.

import sys
import numpy as np


def main():
    if len(sys.argv) != 2:
        print("Error: input parameters error.")
        print(sys.argv[0], "errorArrayFullPath.npy")
        return -1
    errorArrayPath = sys.argv[1] # e.g. `~/temp/muErr_predictR__lmd0_0_4.0_1.0__lmd1_0_4.0_1.0.npy

    errorArray = np.load(errorArrayPath)

    print(f"at both min lambdas, MuError= {errorArray[0,0]}")


if __name__ == "__main__":
    main()