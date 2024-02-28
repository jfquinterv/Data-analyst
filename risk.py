import pandas as pd
import scipy.stats
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize



def annualized_return(rt, periods_per_year):
    """
    Calculate the annualized return.
    
    Parameters:
    rt : array-like
        Array of returns.
    periods_per_year : int
        Number of periods in a year (e.g., 12 for monthly returns).
        
    Returns:
    float
        Annualized return.
    """
    compounded_growth = np.prod(1 + rt)
    n = len(rt)  # using len(rt) as rt may not be a numpy array with shape attribute
    ann_return = compounded_growth ** (periods_per_year / n) - 1
    return ann_return

def annualized_volatility(rt, periods_per_year):
    """
    Calculate the annualized volatility.
    
    Parameters:
    rt : array-like
        Array of returns.
    periods_per_year : int
        Number of periods in a year (e.g., 12 for monthly returns).
        
    Returns:
    float
        Annualized volatility.
    """
    volatility = np.std(rt) * (periods_per_year ** 0.5)
    return volatility

def sharpe_ratio(rt, riskfree_rate, periods_per_year):
    """
    Calculate the Sharpe ratio.
    
    Parameters:
    rt : array-like
        Array of returns.
    riskfree_rate : float
        The risk-free rate per period.
    periods_per_year : int
        Number of periods in a year (e.g., 12 for monthly returns).
        
    Returns:
    float
        Sharpe ratio.
    """
    # Convert the annual risk-free rate to the rate per period
    rf_per_period = (1 + riskfree_rate) ** (1 / periods_per_year) - 1
    excess_return = rt - rf_per_period
    ann_return = annualized_return(excess_return, periods_per_year)
    ann_vol = annualized_volatility(rt, periods_per_year)
    return ann_return / ann_vol
        
    
def drawdown(returns_series: pd.Series):
    """
    Takes a series of assets returns. Computes and returns a data frame that contains:
    - wealth index
    - previous peaks
    - percent drawdowns
    """
    
    wealth_index = 1 * (1 + returns_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks) / previous_peaks
    
    return pd.DataFrame({
        "Wealth": wealth_index,
        "Peaks": previous_peaks,
        "Drawdowns": drawdowns
    })

def semideviation(r):
    "Semi deviation"
    r_negative=r<0
    semideviation=r[r_negative].std(ddof=0)
    return semideviation

def skewness(r):
    "skewness"
    deviation=r-r.mean()
    sigma_3=r.std(ddof=0)
    skewness=(deviation**3).mean()/sigma_3**3
    return skewness

def kurtosis(r):
    """
    Kurtosis
    """
    deviation=r-r.mean()
    sigma=r.std(ddof=0)
    kurtosis=(deviation**4).mean()/sigma**4
    return kurtosis

def normal_test(r, level=0.01):
    """
    Jarque bera test, with defaul significance of 0.05
    """
    statistic, p_value=scipy.stats.jarque_bera(r)
    return p_value>level

def var_historic(r, level=0.05):
    """
    Calculates the Value at Risk (VaR) using the historical method.
    - r: pd.Series or pd.DataFrame representing returns.
    - level: The percentile level to calculate VaR (default 5%).
    Returns VaR values for each column in DataFrame or a single value for Series.
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, level=level)
    elif isinstance(r, pd.Series):
        return -np.percentile(r, level * 100)
    else:
        raise TypeError("Expected r to be a pd.Series or pd.DataFrame")



def var_gaussian(r,level=0.05, modified=False):
    """
   Returns the parametric Gaussian VaR of a series or DataFrame, if modified VaR is returned using the Cornish-Fisher modification.
    """
# Compute the Z score assuming it was gaussian   
    z=norm.ppf(level)
    if modified:
        #modify the z score based on observed skewness and kurtosis
        s=skewness(r)
        k=kurtosis(r)
        z=(z+
           (z**2-1)*s/6+
           (z**3-3*z)*(k-3)/24-
           (2*z**3-5*z)*(s**2)/36
          )
   
    return  -(r.mean()+z*r.std(ddof=0))

def cvar_historic(r, level=0.05):
    """
    Computes the conditional Value at Risk (CVaR) of a series or DataFrame.
    - r: pd.Series or pd.DataFrame representing returns.
    - level: The percentile level to calculate CVaR (default 5%).
    Returns CVaR values for each column in DataFrame or a single value for Series.
    """
    if isinstance(r, pd.Series):
        is_beyond = r <= -var_historic(r, level=level)
        return -r[is_beyond].mean()
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_historic, level=level)
    else:
        raise TypeError("Expected r to be a pd.Series or pd.DataFrame")

    
def semideviation3(r):
    """
    Returns the semideviation aka negative semideviation of r
    r must be a Series or a DataFrame, else raises a TypeError
    """
    excess= r-r.mean()                                       
    excess_negative = excess[excess<0]                      
    excess_negative_square = excess_negative**2               
    n_negative = (excess<0).sum()                             
    return (excess_negative_square.sum()/n_negative)**0.5   

def data_risk():
    """
    Read data in a csv file
    """
    hfi =pd.read_csv('Data/edhec-hedgefundindices.csv',header=0, index_col=0, parse_dates=True, dayfirst=True)
    hfi =hfi/100
    hfi.index=hfi.index.to_period('M')
    return hfi

def data_portafolio():
    """
    Data large cap and small cap
    """
    # Read the CSV file
    rt = pd.read_csv('Data/Portfolios_Formed_on_ME_monthly_EW.csv', header=0,     index_col=0,na_values='-99.99')

    # Select the relevant columns
    rt = rt[['Lo 10', 'Hi 10']]

    # Rename the columns
    rt.columns = ['small cap', 'large cap']

    # Adjust the data for percentage representation
    rt = rt / 100

    # Convert the index to datetime and then to a monthly period
    # Ensure the format matches the actual format of your date column in the CSV file
    rt.index = pd.to_datetime(rt.index, format="%Y%m")
    rt.index = rt.index.to_period('M')

    return rt


def data_ind():
    """
   Ken French 30 Industry portafolio weighted monthly return
    """
    ind=pd.read_csv("Data/ind30_m_vw_rets.csv",header=0,index_col=0)/100
    ind.index=pd.to_datetime(ind.index,format="%Y%m").to_period("M")
    # Columns like Food and Fin has an embedded space, we want to get rid of the embedded space bacause it could cause us problems when we'r manipulating the data in the columns
    ind.columns=ind.columns.str.strip()
    return ind

def portafolio_r(returns,weights):
    """"
    Portafolio returns
    """
    # @ symbol dot product
    return weights.T @ returns

def portafolio_vol(weight,covmatrix):
    """
    weights->volatility
    """
    return (weight.T @ covmatrix @ weight)**0.5


 
def plot_ef2(n_points, er, cov):
    """
    Plots the 2-asset efficient frontier
    """
    if er.shape[0] != 2:
        raise ValueError("plot_ef2 can only plot 2-asset frontiers")
    weights = [np.array([w, 1-w]) for w in np.linspace(0, 1, n_points)]
    rets = [portafolio_r(w, er) for w in weights]
    vols = [portafolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({
        "Returns": rets,
        "Volatility": vols
    })
    return ef.plot.line(x="Volatility", y="Returns", style="--")



def msr(risk_free, er, cov):
    """
    Risk_free_rate+ER+COV->W
    """

    # Number of assets in the portfolio
    n = er.shape[0]

    # Initial guess: Equal weight for each asset (1/n)
    initial_guess = np.repeat(1/n, n)

    # Define the bounds for each weight (between 0 and 1)
    # This creates a tuple of (0.0, 1.0) pairs, one for each asset
    bounds = ((0.0, 1.0),) * n

    # Constraint 1: The sum of weights must be equal to 1
    # This ensures that the total allocated percentage is 100%
    weights_sum_1 = {
        'type': 'eq',
        'fun': lambda weights: np.sum(weights) - 1
    }

    # Optimization function
    # 'options' are additional settings for the optimizer ('disp': False suppresses output)
    # we can embed a function within a function
    # the objetive function is the sharpe ratio, we can maximize using a minimization, minimizing
    # the negative of the objective funtion. ¿how do i define the negative of sharpe ratio?
   
    def neg_sharpe_ratio(weights, risk_free, er, cov):
        """
        return negative sharpe ratio given the weights
        """
        rt=r.portafolio_r(er,weights)
        vol=r.portafolio_vol(weights, cov)
        return -(rt-risk_free)/vol
        
    results = minimize(neg_sharpe_ratio, initial_guess,
                   args=(risk_free, er, cov), method="SLSQP",
                   options={'disp': False},
                   constraints=(weights_sum_1),
                   bounds=bounds)
    # Return the optimized weights
    return results.x

def minimize_vol(target, er, cov):
    """
    Perform portfolio optimization by finding the weights that minimize volatility.

    Parameters:
    - target: The desired return for the portfolio.
    - er: A numpy array containing the expected returns of each asset.
    - cov: The covariance matrix representing the relationships between asset returns.

    Returns:
    - Optimized portfolio weights that meet the specified target return and constraints.
    """

    # Number of assets in the portfolio
    n = er.shape[0]

    # Initial guess: Equal weight for each asset (1/n)
    initial_guess = np.repeat(1/n, n)

    # Define the bounds for each weight (between 0 and 1)
    # This creates a tuple of (0.0, 1.0) pairs, one for each asset
    bounds = ((0.0, 1.0),) * n

    # Constraint 1: Portfolio return should meet the target return
    # 'type' specifies the type of constraint ('eq' for equality)
    # 'args' passes additional required arguments to the function
    # The function calculates the difference between target return and actual return
    # If the constraint is met, this function will return zero
    return_is_target = {
        'type': 'eq',
        'args': (er,), #one element tuple
        'fun': lambda weights, er: target - portafolio_r(er, weights)
    }

    # Constraint 2: The sum of weights must be equal to 1
    # This ensures that the total allocated percentage is 100%
    weights_sum_1 = {
        'type': 'eq',
        'fun': lambda weights: np.sum(weights) - 1
    }

    # Optimization function
    # 'minimize' function is used for finding the minimum of a function
    # 'r.portafolio_vol()' is the function being minimized (portfolio volatility)
    # 'method' specifies the algorithm used for optimization ('SLSQP' for sequential least squares programming)
    # 'options' are additional settings for the optimizer ('disp': False suppresses output)
    # 'constraints' are the defined constraints that the solution must satisfy
    results = minimize(
        lambda weights: portafolio_vol(weights, cov),  # Modified to include the required arguments
        initial_guess,
        method="SLSQP",
        options={'disp': False},
        constraints=(return_is_target, weights_sum_1),
        bounds=bounds
    )

    # Return the optimized weights
    return results.x

def optimal_weights(er,cov, n_points):
    """
    list of weights to run the optimizer on to minimize the vol
    """
    target_rs=np.linspace(er.min(),er.max(),n_points)
    weights=[minimize_vol(target_returns, er, cov) for target_returns in target_rs]
    return weights

def msr(risk_free, er, cov):
    """
   gives us the portafolio with the maximum sharpe ration give the risk free rate, expected returns and the variance-coviariance matrix.
    """

    # Number of assets in the portfolio
    n = er.shape[0]

    # Initial guess: Equal weight for each asset (1/n)
    initial_guess = np.repeat(1/n, n)

    # Define the bounds for each weight (between 0 and 1)
    # This creates a tuple of (0.0, 1.0) pairs, one for each asset
    bounds = ((0.0, 1.0),) * n

    # Constraint 1: The sum of weights must be equal to 1
    # This ensures that the total allocated percentage is 100%
    weights_sum_1 = {
        'type': 'eq',
        'fun': lambda weights: np.sum(weights) - 1
    }

    # Optimization function
    # 'options' are additional settings for the optimizer ('disp': False suppresses output)
    # we can embed a function within a function
    # the objetive function is the sharpe ratio, we can maximize using a minimization, minimizing
    # the negative of the objective funtion. ¿how do i define the negative of sharpe ratio?
   
    def neg_sharpe_ratio(weights, risk_free, er, cov):
        """
        return negative sharpe ratio given the weights
        """
        rt=portafolio_r(er,weights)
        vol=portafolio_vol(weights, cov)
        return -(rt-risk_free)/vol
        
    results = minimize(neg_sharpe_ratio, initial_guess,
                   args=(risk_free, er, cov), method="SLSQP",
                   options={'disp': False},
                   constraints=(weights_sum_1),
                   bounds=bounds)
    # Return the optimized weights
    return results.x
    
def plot_ef(er,cov, n_points, show_cml=False,style='.-',risk_free=0):
    """
    efficient frontier
    """
    weights=optimal_weights(er,cov, n_points)
    rts=[portafolio_r(er,w) for w in weights]
    vols=[portafolio_vol(w,cov)for w in weights]
    ef=pd.DataFrame({"Returns":rts,
                     "Volatility":vols})
    ax=ef.plot.line(x="Volatility", y="Returns", style=style)
    if show_cml:
        ax.set_xlim(left=0)
        w_msr=msr(risk_free,er,cov)
        rt_msr=portafolio_r(er,w_msr)
        vol_msr=portafolio_vol(w_msr,cov)
        # Add CML
        cml_x=[0,vol_msr]
        cml_y=[risk_free,rt_msr]
        ax.plot(cml_x,cml_y,color="green",linestyle="dashed",markersize=10, linewidth=2)
    return ax

