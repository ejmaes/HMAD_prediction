import numpy as np
import pandas as pd
import math

def check_dataframe(df:pd.DataFrame, values:str) -> None:
    if 'time' not in df.columns:
        raise ValueError("No time column in the dataframe.")
    if values not in df.columns:
        raise ValueError("Requested column not in the dataset")

def collapse_intervals(data:pd.DataFrame, collapse_column:str) -> pd.DataFrame:
    """collapsing adjacent intervals with same label into one
    """
    df = data.groupby((data[collapse_column] != data[collapse_column].shift()).cumsum()).agg(**{
        'tmin':pd.NamedAgg(column='tmin', aggfunc= min),
        'tmax':pd.NamedAgg(column='tmax', aggfunc= max),
        collapse_column:pd.NamedAgg(column=collapse_column, aggfunc= pd.unique)
    }).reset_index(drop=True)
    return df

################## from UEDialogueSRsources.R
### Computation of wavelets
def computePredictor2Value(times: np.array, values: np.array, duration: float, starts_down:bool=True, **kwargs):
    """
    Compute the goodness of fit and the amplitude of the values for a given range at a given index
    
    Input:
    -------
    times: np.array
    values: np.array
        filtered column in data - The values of the signal at relevant times
    duration: float 
    starts_down:bool
        Boolean at TRUE if the basis function starts to go down before going up
    kwargs: 
        other parameters (index, range, n) if using original computation (debug)

    Return :
    -------
        a vector with wgof and wcoef
    """
    # using time instead of indexing
    """
    """
    if kwargs != {}:
        index = kwargs['index']
        rg = kwargs['range']
        n = kwargs['n']
        ## index min and index max (accounting for the boundaries of the arrays
        imin = int(max(0, index - rg))
        imax = int(min(n-1, index + rg))
        ## The effective range of the interval, 2*range if out of the boudaries
        effective_range = imax - imin + 1
        ## computing w
        w = np.array([np.sin(2*math.pi*(i-imin)/(imax-imin)) for i in range(imin, imax+1)])
        values = values.to_numpy()[imin:imax+1]
    else:
        w = np.sin(2*math.pi*(times-duration/2)/(duration))
        effective_range = len(times)
    
    #sf2 = np.dot(values,values) 
    #sfw = np.dot(values,w)
    # with normalisation
    vmean = values.mean()
    sf2 = np.dot(values - vmean,values - vmean) 
    sw2 = np.dot(w,w)
    sfw = np.dot(values - vmean,w)

    ## Normalize
    if sf2 == 0:
        gof = 0
    else:
        gof = sfw / np.sqrt(sf2*sw2)
    coefficient = sfw / effective_range

    ## The result as a couple goodness-of-fit and coefficent
    ## Change the sign of the result of the option down is TRUE
    if starts_down:
        return (-gof, -coefficient)
    return (gof, coefficient)


def computeArrayPredictor2Value(data:pd.DataFrame, values:str, duration:float, starts_down:bool=True, use_index:bool=False) -> pd.DataFrame:
    """
    Compute the goodness of fit and the amplitude of the values for a given basis function duration
    
    Input:
    -------
    data: pd.DataFrame
        dataframe with The times column
    values: str
        Column name - The values of the signal at times
    down: bool
        TRUE if the basis function starts to go down before going up
    duration: float
        The duration (in s)
    use_index: bool
        whether to use the original method (debug) or an optimized method
    
    Return 
    --------
    prediction: pd.DataFrame
        dataframe with wgof and wcoef
    """
    wgof = np.empty(data.shape[0])
    wcoef = np.empty(data.shape[0])
    for i, time_inst in enumerate(data.time.tolist()):
        time_idx = np.where(np.logical_and(data.time >= time_inst-duration/2, data.time <= time_inst+duration/2))[0]

        if use_index:
            wgof[i], wcoef[i] = computePredictor2Value(data, data[values], duration, starts_down=starts_down, range=np.round(duration * 25 / 2), index = i, n = data.shape[0])
        else:
            wgof[i], wcoef[i] = computePredictor2Value(data.time[time_idx], data[values][time_idx], duration, starts_down=starts_down)
    
    predictions = data[['time', values]]
    predictions['wgof'] = wgof
    predictions['wcoef'] = wcoef
    return predictions


def compute2DArrayPredictor2Value(data:pd.DataFrame, values:str, sigmamin:float, sigmamax:float, sigmastep:float, starts_down:bool=True, use_index:bool=True) -> pd.DataFrame:
    """masco2020.sr.compute2DArrayPredictor2Value
    Compute the goodness of fit and the amplitude of the values on a grid defined by igmamin, sigmamax and sigmastep

    Input
    ---------
    data: pd.DataFrame
        columns: ['times', values, ...]
    values:str
        The values of the signal at times
    sigmamin: float
        The minimal duration of the grid
    sigmamax: float
        The maximal duration of the grid
    sigmastep: float
        The step duration of the grid
    down: bool 
        Boolean at TRUE if the basis function starts to go down before going up

    Return 
    ---------
        a data frame with times, duration, wgof and wcoef
    """
    res = []

    # For each time_scale, create a dataframe containing wgof and wcoef coefficients
    rg = np.around(np.arange(sigmamin,sigmamax,sigmastep), decimals=4) # floating point issues
    if (round(sigmamax - rg[-1],5) == sigmastep): # floating point issues
        rg = np.append(rg, sigmamax) # append last value if necessary

    for i, time_scale in enumerate(rg):
        tmp = computeArrayPredictor2Value(data, values, time_scale, starts_down=starts_down, use_index=use_index)
        # contains time, wgof and wcoef
        tmp['duration'] = time_scale #!!! time period of wavelet 
        res.append(tmp)
        print(f"Scale {i+1} completed, timescale = {time_scale}s.")
    
    res = pd.concat(res, ignore_index=True)
    return res.applymap(lambda x: np.round(x,4))

### Used for prediction
def computeMaximalTimeIntervals(df:pd.DataFrame, 
                                interest_col:str = 'wgof'):
    """
    Compute the set of maximal intervals:
    Starting with intervals with the strongest certainty
    * Comparing start / end of interval to current interval;
    * If interval overlaps with current interval, remove linie from df

    Input:
    -------
    df: pd.DataFrame
        shape ['time', feature, 'wgof', 'wcoef', 'duration']
    
    interest_col: str 
        The index of the column for the value to maximize in the data frame df
    
    Return:
    -------
        a list of time intervals (not adjacent)
    """
    ## boundaries for each timestep/duration pair 
    df['tmin'] = df.time - df.duration/2
    df['tmax'] = df.time + df.duration/2

    ## removing overlap intervals (partial overlap is enough)
    # initialisation - Sort the table by decreasing order
    dfsd = df[(df.tmin >= 0) & (df.tmax <= df.time.max())].sort_values(by=interest_col, ascending=False)
    res = [] # storing rows of interest
    #for idx, row in df.iterrows(): - while loop more memory friendly
    while dfsd.shape[0] > 0:
        current_row = dfsd.iloc[0]
        dfsd = dfsd[ (dfsd.tmin >= current_row.tmax) | (dfsd.tmax <= current_row.tmin) ]
        # removing overlapping rows also removes the current row => stored
        res.append(current_row)
    
    return pd.DataFrame(res).sort_values(by=['tmin', 'time'])


### Prediction
def fillTimeColumnInterval(res:pd.DataFrame, time_start = None, time_stop = None, other_columns:dict = {}):
    """
    Fill the missing intervals

    Input:
    ------
    res: pd.DataFrame
        containts `tmin` and `tmax` columns
    
    time_start, time_stop: float
        whether to add first and last line
    
    other_columns: dict
        other values to add to the created columns (default value)
    """
    if ('tmin' not in res.columns) or ('tmax' not in res.columns):
        raise ValueError("`tmin` and `tmax` values must be included in dataset.")

    res['tmax_prev'] = res.tmax.shift(1)
    if time_start is not None:
        res['tmax_prev'] = res['tmax_prev'].fillna(time_start)
    res['time_since_prev'] = np.round(res.tmin - res.tmax_prev, 4) # rounding issues
    blank_rows = [{
        'tmin': row.tmax_prev,
        'tmax': row.tmin,
        'time': (row.tmax_prev + row.tmin)/2,
        'duration': row.time_since_prev
    } for _, row in res[res.time_since_prev > 0].iterrows()]
    # adding last row
    if time_stop is not None:
        blank_rows.append({
            'tmin': res.tmax_prev.iloc[-1],
            'tmax': time_stop,
            'time': (res.tmax_prev.iloc[-1] + time_stop)/2,
            'duration': (res.tmax_prev.iloc[-1] - time_stop)/2
        })

    tmp = pd.DataFrame(blank_rows)
    for col, defaultvalue in other_columns.items():
        tmp[col] = defaultvalue
    res = res.append(tmp).sort_values(by='tmin').reset_index(drop=True)
    return res

def createNodAnnotations(df, values, nod_threshold, 
                        smin=0.24, smax=1.04, step=0.08,
                        use_index=False, dresults=None):
    """
    Create the automatic nod annotations predicted from the HMAD output data

    Input:
    -------
    df: pd.DataFrame
        The output data frame created by HMAD
    
    values: str
        The ouput column to consider
    
    nod_threshold: float
        The threshold in the result value to select nod and no_nod annotation
    
    (smin, smax, step): float
        limits for the 2 dimensional space for the search
    
    dresults: pd.DataFrame
        if results from the 2D wavelets have already been computed, no need to do it again

    Return:
    -------
        a data frame including tmin tmax time intervals with annotations nod or no_nod (column nod_predictions)
    """
    check_dataframe(df, values)
    if dresults is None:
        dresults = compute2DArrayPredictor2Value(df, values, smin, smax, step, use_index=use_index)
    # removing overlaping intervals
    res = computeMaximalTimeIntervals(dresults)
    # filling in with missing intervals
    res = fillTimeColumnInterval(res)

    # Predictions based on nod_threshold
    res['prediction'] = res.wgof.apply(lambda x: "NA" if np.isnan(x) else ("nod" if x >= nod_threshold else "no_nod"))
    # Thumb Rule here - if between two nods then is nod
    res['before_after'] = pd.concat([   res['prediction'].shift(1).fillna("OUT"), 
            res['prediction'].shift(-1).fillna("OUT")
        ], axis=1).values.tolist()
    res['prediction'] = res.apply(lambda x: x.prediction if x.prediction != "NA" else (x.before_after[0] if x.before_after[0] == x.before_after[1] else "no_nod"), axis=1)

    return res[['tmin', 'tmax', values, 'wgof', 'wcoef', 'prediction']]

def projectTimeIntervalsCharValues(data:pd.DataFrame, nod_inflate:pd.DataFrame, column_name:str="prediction"):
    """Project on the times line the [tmin, tmax] time intervals for the values the given column index.
    Can be used for gold standard as well as prediction
    """
    data[column_name] = "no_nod"
    for idx, row in nod_inflate[nod_inflate[column_name] == 'nod'].iterrows():
        nod_idx = np.where((data.time >= row.tmin) & (data.time <= row.tmax))[0]
        data[column_name].iloc[nod_idx] = "nod"
    return data
