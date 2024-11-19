# Axivity Outdoor Light Exposure Analysis

## Install

*Minimum requirements*: Python 3.8 to 3.10, Java 8 (1.8)

The following instructions make use of Anaconda to meet the minimum requirements:

1. Download & install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (light-weight version of Anaconda).
1. (Windows) Once installed, launch the **Anaconda Prompt**.
1. Create a virtual environment (here named `axlux`):
    ```console
    $ conda create -n axlux python=3.9 openjdk pip
    ```
    This creates a virtual environment called `axlux` with Python version 3.9, OpenJDK, and Pip.
1. Activate the environment:
    ```console
    $ conda activate axlux
    ```
    You should now see `(axlux)` written in front of your prompt.
1. Install the `axivity-outdoor-light` package:
    - Method 1 (SSH):
    ```console
    $ pip install 'axivity-outdoor-light @ git+ssh://git@github.com/OxWearables/axivity-outdoor-light'
    ```
    - Method 2 (HTTPS):
    ```console
    $ pip install 'axivity-outdoor-light @ git+https://github.com/OxWearables/axivity-outdoor-light'
    ```

You are all set! The next time that you want to use `axlux`, open the Anaconda Prompt and activate the environment (step 4). If you see `(axlux)` in front of your prompt, you are ready to go!

## Usage

```bash
# Process an AX3 file
$ axlux sample.cwa

# Or an ActiGraph file
$ axlux sample.gt3x

# Or a GENEActiv file
$ axlux sample.bin

# Or a CSV file (see accepted data format below)
$ axlux sample.csv
```

Output:
```console
Summary
-------
{
    "Filename": "sample.cwa",
    "Filesize(MB)": 69.4,
    "Device": "Axivity",
    "DeviceID": 13110,
    "ReadErrors": 0,
    "SampleRate": 100.0,
    "ReadOK": 1,
    "StartTime": "2014-05-07 13:29:50",
    "EndTime": "2014-05-13 09:50:33",
    "TotalOutdoorLight(mins)": 492.0,
    "OutdoorLightDayAvg(mins)": 70.28571428571429,
    "OutdoorLightDayMed(mins)": 51.0,
    "OutdoorLightDayMin(mins)": 12.0,
    "OutdoorLightDayMax(mins)": 165.0,
    ...
}

Estimated Daily Outdoor Light
-----------------------------
            OutdoorLight(mins)  OutdoorLightAdjusted(mins)
time
2014-05-07            51.0               63.166667
2014-05-08            39.0               39.000000
2014-05-09            68.0               68.000000
2014-05-10           120.0              120.000000
...

Output: outputs/sample/
```

### Troubleshooting 
Some systems may face issues with Java when running the script. If this is your case, try fixing OpenJDK to version 8:
```console
$ conda install -n axlux openjdk=8
```

### Output files
By default, output files will be stored in a folder named after the input file, `outputs/{filename}/`, created in the current working directory. You can change the output path with the `-o` flag:

```console
$ axlux sample.cwa -o /path/to/some/folder/
```

The following output files are created:

- *Info.json* Summary info, as shown above.
- *Minutes.csv* Raw time-series of outdoor light minute-by-minute estimates.
- *Hourly.csv* Hourly outdoor light time.
- *Daily.csv* Daily outdoor light time. 
- *HourlyAdjusted.csv* Like Hourly but accounting for missing data (see section below).
- *DailyAdjusted.csv* Like Daily but accounting for missing data (see section below).

### Crude vs. Adjusted Estimates
Adjusted estimates are provided that account for missing data.
Missing values in the time-series are imputed with the mean of the same timepoint of other available days.
For adjusted totals and daily statistics, 24h multiples are needed and will be imputed if necessary.
Estimates will be NaN where data is still missing after imputation.

### Processing CSV files
TODO

## Licence
TODO

## Acknowledgements
We would like to thank all our code contributors, manuscript co-authors, and research participants for their help in making this work possible.
