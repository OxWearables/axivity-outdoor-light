import os
import warnings
import time
import datetime as dt
import argparse
import pathlib
import json
import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset
import scipy.stats as stats
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

import actipy



def main():

    before = time.time()

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("filepath", help="Enter file to be processed")
    parser.add_argument("--outdir", "-o", help="Enter folder location to save output files", default="outputs/")
    # parser.add_argument("--natol", "-t", help="Enter NA tolerance", default=0.2)
    args = parser.parse_args()

    basename = resolve_path(args.filepath)[1]
    outdir = os.path.join(args.outdir, basename)
    os.makedirs(outdir, exist_ok=True)

    print(f"Processing file: {args.filepath}")

    data, info = actipy.read_device(
        args.filepath,
        lowpass_hz=None,
        calibrate_gravity=True,
        detect_nonwear=True,
        resample_hz=None,
    )

    # ENMO
    data['enmo'] = 1000 * np.maximum(np.linalg.norm(data[['x', 'y', 'z']], axis=1) - 1, 0)

    # Drop no longer needed columns
    data = data.drop(columns=['x', 'y', 'z', 'temperature'])

    # Minute averages
    data = data.resample('1min').mean()

    # Mean ENMO
    info['ENMO(mg)'] = impute_missing(data['enmo']).mean()

    sunlit = pd.Series(index=data.index, dtype='float')

    # for t, g in data.groupby((data.index - pd.Timedelta(hours=10)).date):
    #     na = g.isna().any(axis=1)
    #     # check if there is enough data
    #     if len(g[~na]) < (1 - tol) * 1440:  # 1440 minutes in a day
    #         print(f"Skipping {t} due to insufficient data")
    #         continue
    #     sunlit.loc[g[~na].index] = predict_outdoor(g.loc[~na, 'light'])

    print("Estimating outdoor light...")
    na = data.isna().any(axis=1)
    sunlit.loc[~na] = predict_outdoor(data.loc[~na, 'light'])

    # rolling mode smoothing
    def mode(x):
        if len(x) == 0:
            return np.nan
        if np.isnan(x[-1]):
            return np.nan
        return stats.mode(x, nan_policy='omit')[0]

    sunlit = sunlit.rolling(5, min_periods=1).apply(mode, raw=True)

    fig = plot(sunlit, enmo=data['enmo'], label='Outdoor Light', title=basename)
    fig.savefig(f"{outdir}/{basename}-Plot.png")

    # Moderate to Vigorous Physical Activity
    mvpa = (data['enmo'] >= 100).astype('float')  # 100 millig
    mvpa[na] = np.nan

    # Moderate to Vigorous Physical Activity and Outdoor Light
    sunlit_and_mvpa = (sunlit.astype('bool') & mvpa.astype('bool')).astype('float')
    sunlit_and_mvpa[na] = np.nan

    pd.concat([
        sunlit.rename('Outdoor'),
        mvpa.rename('MVPA'),
        sunlit_and_mvpa.rename('OutdoorMVPA')
    ], axis=1).to_csv(f"{outdir}/{basename}-Minutes.csv")

    # Summary
    sunlit_summa = summarize(sunlit, adjust_estimates=False)
    info['TotalOutdoorLight(mins)'] = sunlit_summa['total']
    info['OutdoorLightDayAvg(mins)'] = sunlit_summa['daily_avg']
    info['OutdoorLightDayMed(mins)'] = sunlit_summa['daily_med']
    info['OutdoorLightDayMin(mins)'] = sunlit_summa['daily_min']
    info['OutdoorLightDayMax(mins)'] = sunlit_summa['daily_max']

    mvpa_summa = summarize(mvpa, adjust_estimates=False)
    info['TotalMVPA(mins)'] = mvpa_summa['total']
    info['MVPADayAvg(mins)'] = mvpa_summa['daily_avg']
    info['MVPADayMed(mins)'] = mvpa_summa['daily_med']
    info['MVPADayMin(mins)'] = mvpa_summa['daily_min']
    info['MVPADayMax(mins)'] = mvpa_summa['daily_max']

    sunlit_and_mvpa_summa = summarize(sunlit_and_mvpa, adjust_estimates=False)
    info['TotalOutdoorMVPA(mins)'] = sunlit_and_mvpa_summa['total']
    info['OutdoorMVPADayAvg(mins)'] = sunlit_and_mvpa_summa['daily_avg']
    info['OutdoorMVPADayMed(mins)'] = sunlit_and_mvpa_summa['daily_med']
    info['OutdoorMVPADayMin(mins)'] = sunlit_and_mvpa_summa['daily_min']
    info['OutdoorMVPADayMax(mins)'] = sunlit_and_mvpa_summa['daily_max']

    pd.concat([
        sunlit_summa['daily'].rename('OutdoorLight(mins)'),
        mvpa_summa['daily'].rename('MVPA(mins)'),
        sunlit_and_mvpa_summa['daily'].rename('OutdoorMVPA(mins)')
    ], axis=1).to_csv(f"{outdir}/{basename}-Daily.csv")

    pd.concat([
        sunlit_summa['hourly'].rename('OutdoorLight(mins)'),
        mvpa_summa['hourly'].rename('MVPA(mins)'),
        sunlit_and_mvpa_summa['hourly'].rename('OutdoorMVPA(mins)')
    ], axis=1).to_csv(f"{outdir}/{basename}-Hourly.csv")

    # Summary Adjusted
    sunlit_summa_adj = summarize(sunlit, adjust_estimates=True)
    info['TotalOutdoorLightAdjusted(mins)'] = sunlit_summa_adj['total']
    info['OutdoorLightDayAvgAdjusted(mins)'] = sunlit_summa_adj['daily_avg']
    info['OutdoorLightDayMedAdjusted(mins)'] = sunlit_summa_adj['daily_med']
    info['OutdoorLightDayMinAdjusted(mins)'] = sunlit_summa_adj['daily_min']
    info['OutdoorLightDayMaxAdjusted(mins)'] = sunlit_summa_adj['daily_max']

    mvpa_summa_adj = summarize(mvpa, adjust_estimates=True)
    info['TotalMVPAAdjusted(mins)'] = mvpa_summa_adj['total']
    info['MVPADayAvgAdjusted(mins)'] = mvpa_summa_adj['daily_avg']
    info['MVPADayMedAdjusted(mins)'] = mvpa_summa_adj['daily_med']
    info['MVPADayMinAdjusted(mins)'] = mvpa_summa_adj['daily_min']
    info['MVPADayMaxAdjusted(mins)'] = mvpa_summa_adj['daily_max']

    sunlit_and_mvpa_summa_adj = summarize(sunlit_and_mvpa, adjust_estimates=True)
    info['TotalOutdoorMVPAAdjusted(mins)'] = sunlit_and_mvpa_summa_adj['total']
    info['OutdoorMVPADayAvgAdjusted(mins)'] = sunlit_and_mvpa_summa_adj['daily_avg']
    info['OutdoorMVPADayMedAdjusted(mins)'] = sunlit_and_mvpa_summa_adj['daily_med']
    info['OutdoorMVPADayMinAdjusted(mins)'] = sunlit_and_mvpa_summa_adj['daily_min']
    info['OutdoorMVPADayMaxAdjusted(mins)'] = sunlit_and_mvpa_summa_adj['daily_max']

    pd.concat([
        sunlit_summa_adj['daily'].rename('OutdoorLightAdjusted(mins)'),
        mvpa_summa_adj['daily'].rename('MVPAAdjusted(mins)'),
        sunlit_and_mvpa_summa_adj['daily'].rename('OutdoorMVPAAdjusted(mins)')
    ], axis=1).to_csv(f"{outdir}/{basename}-DailyAdjusted.csv")

    pd.concat([
        sunlit_summa_adj['hourly'].rename('OutdoorLightAdjusted(mins)'),
        mvpa_summa_adj['hourly'].rename('MVPAAdjusted(mins)'),
        sunlit_and_mvpa_summa_adj['hourly'].rename('OutdoorMVPAAdjusted(mins)')
    ], axis=1).to_csv(f"{outdir}/{basename}-HourlyAdjusted.csv")

    # Save info
    with open(f"{outdir}/{basename}-Info.json", 'w') as f:
        json.dump(info, f, indent=4, cls=NpEncoder)

    # Print
    print("\nSummary\n-------")
    print(json.dumps(info, indent=4, cls=NpEncoder))
    print("\nEstimated Daily OutdoorLight\n---------------------")
    print(pd.concat([
        sunlit_summa['daily'].rename('OutdoorLight(mins)'),
        sunlit_summa_adj['daily'].rename('OutdoorLightAdjusted(mins)'),
    ], axis=1))

    print(f"Done! ({time.time() - before:.2f}s)")
    print(f"Output files saved in: {outdir}/")


def predict_outdoor(x, n_clusters=4):
    if isinstance(x, (pd.Series, pd.DataFrame)):
        x = x.to_numpy()
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, copy_x=False)
    yc = kmeans.fit_predict(np.log(1 + x))
    if len(np.unique(yc)) < n_clusters:
        warnings.warn(f"KMeans only found {len(np.unique(yc))} clusters")
        return np.zeros_like(yc).astype('int')
    max_id = np.argmax(kmeans.cluster_centers_)
    y = (yc == max_id).astype('int')
    return y


def summarize(Y, adjust_estimates=False):

    if adjust_estimates:
        Y = impute_missing(Y)
        skipna = False
    else:
        # crude summary ignores missing data
        skipna = True

    def _sum(x):
        x = x.to_numpy()
        if skipna:
            return np.nansum(x)
        return np.sum(x)

    # there's a bug with .resample().sum(skipna)
    # https://github.com/pandas-dev/pandas/issues/29382

    total = Y.agg(_sum)
    hourly = Y.resample('H').agg(_sum)
    daily = Y.resample('D').agg(_sum)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*Mean of empty slice.*")
        daily_med = daily.median()
    daily_avg = daily.mean()
    daily_min = daily.min()
    daily_max = daily.max()

    return {
        'total': total,
        'hourly': hourly,
        'daily': daily,
        'daily_avg': daily_avg,
        'daily_med': daily_med,
        'daily_min': daily_min,
        'daily_max': daily_max,
    }


def impute_missing(data: pd.DataFrame, extrapolate=True):

    if extrapolate:
        # padding at the boundaries to have full 24h
        data = data.reindex(
            pd.date_range(
                data.index[0].floor('D'),
                data.index[-1].ceil('D'),
                freq=to_offset(infer_freq(data.index)),
                inclusive='left',
                name='time',
            ),
            method='nearest',
            tolerance=pd.Timedelta('1m'),
            limit=1)

    def fillna(subframe):
        if isinstance(subframe, pd.Series):
            x = subframe.to_numpy()
            na = np.isnan(x)
            nanlen = len(x[na])
            if 0 < nanlen < len(x):  # check x contains a NaN and is not all NaN
                x[na] = np.nanmean(x)
                return x  # will be cast back to a Series automatically
            else:
                return subframe

    data = (
        data
        # first attempt imputation using same day of week
        .groupby([data.index.weekday, data.index.hour, data.index.minute])
        .transform(fillna)
        # then try within weekday/weekend
        .groupby([data.index.weekday >= 5, data.index.hour, data.index.minute])
        .transform(fillna)
        # finally, use all other days
        .groupby([data.index.hour, data.index.minute])
        .transform(fillna)
    )

    return data


def infer_freq(x):
    """ Like pd.infer_freq but more forgiving """
    freq, _ = stats.mode(np.diff(x), keepdims=False)
    freq = pd.Timedelta(freq)
    return freq


def nanint(x):
    if np.isnan(x):
        return x
    return int(x)


def resolve_path(path):
    """ Return parent folder, file name and file extension """
    p = pathlib.Path(path)
    extension = p.suffixes[0]
    filename = p.name.rsplit(extension)[0]
    dirname = p.parent
    return dirname, filename, extension


def plot(data, enmo=None, label=None, title=None):
    """ Plot outdoor for each day """

    grouped = data.groupby(data.index.date)
    nrows = len(grouped) + 1

    # setup plotting range
    if enmo is not None:
        enmo = enmo.clip(0, 2000) / 2000  # normalize (max 2g)

    fig = plt.figure(None, figsize=(10, nrows), dpi=100)

    # plot each day
    i = 0
    axs = []
    for day, group in grouped:

        ax = fig.add_subplot(nrows, 1, i + 1)

        if enmo is not None:
            ax.plot(group.index, enmo.loc[group.index].to_numpy(), c='k')

        ax.stackplot(group.index, group.to_numpy(), edgecolor="none")

        # add date label to left hand side of each day's activity plot
        ax.set_ylabel(day.strftime("%A\n%d %B"),
                      weight='bold',
                      horizontalalignment='right',
                      verticalalignment='center',
                      rotation='horizontal',
                      fontsize='medium',
                      color='k',
                      labelpad=5)
        # run gridlines for each hour bar
        ax.get_xaxis().grid(True, which='major', color='grey', alpha=1.00)
        ax.get_xaxis().grid(True, which='minor', color='grey', alpha=0.25)
        # set x and y-axes
        ax.set_xlim(group.index[0], group.index[-1])
        ax.set_xticks(pd.date_range(start=dt.datetime.combine(day, dt.time(0, 0, 0, 0)),
                                    end=dt.datetime.combine(day + dt.timedelta(days=1), dt.time(0, 0, 0, 0)),
                                    freq='4H'))
        ax.set_xticks(pd.date_range(start=dt.datetime.combine(day, dt.time(0, 0, 0, 0)),
                                    end=dt.datetime.combine(day + dt.timedelta(days=1), dt.time(0, 0, 0, 0)),
                                    freq='1H'), minor=True)
        ax.set_ylim(0, 1)
        ax.get_yaxis().set_ticks([])  # hide y-axis lables
        # make border less harsh between subplots
        ax.spines['top'].set_color('#d3d3d3')  # lightgray
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        # set background colour to lightgray
        ax.set_facecolor('#d3d3d3')

        # append to list and incrament list counter
        axs.append(ax)
        i += 1

    # create new subplot to display legends
    ax = fig.add_subplot(nrows, 1, i + 1)
    ax.axis('off')
    legend_patches = [mlines.Line2D([], [], color='k', label='acceleration')]
    legend_patches.append(mpatches.Patch(color='C0', label=label))
    # create overall legend
    plt.legend(handles=legend_patches, bbox_to_anchor=(0., 0., 1., 1.),
               loc='center', ncol=4, mode="best",
               borderaxespad=0, framealpha=0.6, frameon=True, fancybox=True)

    # remove legend border
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    axs.append(ax)

    # format x-axis to show hours
    fig.autofmt_xdate()
    # add hour labels to top of plot
    hrLabels = ['00:00', '04:00', '08:00', '12:00', '16:00', '20:00', '24:00']
    axs[0].set_xticklabels(hrLabels)
    axs[0].tick_params(labelbottom=False, labeltop=True, labelleft=False)

    # add title
    fig.suptitle(title)

    # auto trim borders
    fig.tight_layout()

    return fig


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)



if __name__ == '__main__':
    main()
