import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import integrate
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class BatesModelAnalyzer:
    def __init__(self, S0, r, V0, kappa, theta, epsilon, rho, 
                 lambda_jump, mu_jump, sigma_jump):
        # Input validation and conversion
        self.validate_inputs(S0, r, V0, kappa, theta, epsilon, rho, 
                           lambda_jump, mu_jump, sigma_jump)
        
        self.S0 = float(S0)
        self.r = float(r)
        self.V0 = float(V0)
        self.kappa = float(kappa)
        self.theta = float(theta)
        self.epsilon = float(epsilon)
        self.rho = float(rho)
        self.lambda_jump = float(lambda_jump)
        self.mu_jump = float(mu_jump)
        self.sigma_jump = float(sigma_jump)
        self.m = self.compute_m()

    def validate_inputs(self, S0, r, V0, kappa, theta, epsilon, rho, 
                       lambda_jump, mu_jump, sigma_jump):
       
        if S0 <= 0 or V0 <= 0 or kappa <= 0 or theta <= 0 or epsilon <= 0:
            raise ValueError("S0, V0, kappa, theta, and epsilon must be positive")
        if abs(rho) >= 1:
            raise ValueError("rho must be between -1 and 1")
        if lambda_jump < 0:
            raise ValueError("lambda_jump must be non-negative")

    def compute_m(self):

        try:
            return np.exp(self.mu_jump + 0.5 * self.sigma_jump**2) - 1
        except OverflowError:
            return np.sign(self.mu_jump) * 1e8

    def safe_sqrt(self, x):

        if isinstance(x, complex):
            return np.sqrt(complex(max(x.real, 0), x.imag))
        return np.sqrt(max(x, 0))

    def safe_exp(self, x, threshold=100):

        if isinstance(x, complex):
            real_part = min(max(x.real, -threshold), threshold)
            return np.exp(complex(real_part, x.imag))
        return np.exp(min(max(x, -threshold), threshold))

    def safe_log(self, x):
     
        if isinstance(x, complex):
            if abs(x) < 1e-15:
                return complex(-50, 0)
            return np.log(x)
        return np.log(max(x, 1e-15))

    def characteristic_function(self, u, T):
 
        try:
            u = complex(float(u.real), float(u.imag))
            a = self.kappa * self.theta
            b = self.kappa
            rho_epsilon_u = self.rho * self.epsilon * u * 1j
            
            # Compute d 
            d_squared = (rho_epsilon_u - b)**2 + self.epsilon**2 * (u**2 + u * 1j)
            d = self.safe_sqrt(d_squared)
            
            # Compute g 
            denominator = b - rho_epsilon_u + d
            if abs(denominator) < 1e-15:
                g = 0
            else:
                g = (b - rho_epsilon_u - d) / denominator
            
            # Time component
            exp_dt = self.safe_exp(-d * T)
            g_exp = g * exp_dt

            if abs(1 - g) < 1e-15 or abs(1 - g_exp) < 1e-15:
                log_term = 0
            else:
                log_term = self.safe_log((1 - g_exp) / (1 - g))
   
            C = (self.r - self.lambda_jump * self.m) * u * 1j * T
            
            D = (a / (self.epsilon**2)) * (
                (b - rho_epsilon_u - d) * T - 2 * log_term
            )
            
            if abs(1 - g_exp) < 1e-15:
                E = 0
            else:
                E = ((b - rho_epsilon_u - d) / self.epsilon**2) * \
                    (1 - exp_dt) / (1 - g_exp)
            
            # Jump component 
            jump_term = self.mu_jump * u * 1j + 0.5 * self.sigma_jump**2 * u**2
            jump_exp = self.safe_exp(jump_term)
            jump_comp = self.lambda_jump * T * (jump_exp - 1)
        
            exponent = C + D + E * self.V0 + jump_comp
            return self.safe_exp(exponent)
            
        except Exception as e:
            return complex(1e-50, 0)

    def price_european_option(self, K, T, option_type='call'):
        """Price European option with integration method"""
        k = np.log(K / self.S0)
        
        def integrand(u):
            try:
                if option_type == 'call':
                    shift = -0.5j
                else:
                    shift = 0.5j
                
                phi = self.characteristic_function(u + shift, T)
                denominator = u**2 + 0.25
                
                if abs(denominator) < 1e-15:
                    return 0.0
                
                result = np.exp(-1j * u * k) * phi / denominator
                return float(result.real)
                
            except Exception:
                return 0.0

    
        try:
            integration_result = integrate.quad(
                integrand,
                0,
                50, 
                limit=1000,
                epsabs=1e-8,
                epsrel=1e-8,
                points=[0.1, 1.0, 10.0]  
            )[0]
            
            if option_type == 'call':
                price = self.S0 * 0.5 + self.S0 * integration_result / np.pi
            else:
                price = self.S0 * 0.5 - self.S0 * integration_result / np.pi
            
            price = price * np.exp(-self.r * T)
            
            # Ensure no-arbitrage bounds
            if option_type == 'call':
                lower_bound = max(0, self.S0 * np.exp(-self.r * T) - K * np.exp(-self.r * T))
                upper_bound = self.S0
                return max(lower_bound, min(price, upper_bound))
            else:
                lower_bound = max(0, K * np.exp(-self.r * T) - self.S0 * np.exp(-self.r * T))
                upper_bound = K * np.exp(-self.r * T)
                return max(lower_bound, min(price, upper_bound))
                
        except Exception:
            if option_type == 'call':
                return max(0, self.S0 - K * np.exp(-self.r * T))
            else:
                return max(0, K * np.exp(-self.r * T) - self.S0)
    
class OptionAnalysis:
    def __init__(self, data_df):
        self.data = data_df.copy()
        self.prepare_data()
        
    def prepare_data(self):
        self.data['datetime'] = pd.to_datetime(self.data['datetime'])
        self.data['expiry_date'] = pd.to_datetime(self.data['expiry_date'])

    def calculate_theoretical_prices(self):
        """Calculate theoretical prices using Bates model"""
        model = BatesModelAnalyzer(
            S0=self.data['close_spot'].mean(),
            r=0.05,  # Assumed risk-free rate
            V0=self.data['sigma_20'].mean()**2,
            kappa=2.0,
            theta=0.04,
            epsilon=0.3,
            rho=-0.5,
            lambda_jump=1.0,
            mu_jump=-0.1,
            sigma_jump=0.2
        )
        
        theoretical_prices = []
        for _, row in self.data.iterrows():
            if pd.isna(row['close_option']):
                theoretical_prices.append(np.nan)
                continue
                
            price = model.price_european_option(
                K=row['strike_price'],
                T=row['time_to_expiry'],
                option_type=row['right'].lower()
            )
            theoretical_prices.append(price)
            
        self.data['theoretical_price'] = theoretical_prices
        return theoretical_prices

    def calculate_mse(self):

        mask = ~(self.data['theoretical_price'].isna() | 
                self.data['close_option'].isna())
        
        mse = np.mean((self.data.loc[mask, 'theoretical_price'] - 
                      self.data.loc[mask, 'close_option'])**2)
        rmse = np.sqrt(mse)
        
        return {'mse': mse, 'rmse': rmse}

    def plot_all_analysis(self):

        fig = plt.figure(figsize=(20, 15))
        
        # 1. Price Comparison Time Series
        ax1 = fig.add_subplot(321)
        ax1.plot(self.data['datetime'], self.data['close_option'], 
                label='Actual', marker='o')
        ax1.plot(self.data['datetime'], self.data['theoretical_price'], 
                label='Theoretical', marker='x')
        ax1.set_title('Option Price Comparison')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True)
        plt.setp(ax1.xaxis.get_ticklabels(), rotation=45)
        
        # 2. Price Difference Distribution
        ax2 = fig.add_subplot(322)
        price_diff = self.data['theoretical_price'] - self.data['close_option']
        sns.histplot(price_diff.dropna(), kde=True, ax=ax2)
        ax2.set_title('Price Difference Distribution')
        ax2.set_xlabel('Price Difference (Theoretical - Actual)')
        ax2.set_ylabel('Frequency')
        
        # 3. Scatter Plot of Actual vs Theoretical
        ax3 = fig.add_subplot(323)
        ax3.scatter(self.data['close_option'], self.data['theoretical_price'])
        min_val = min(self.data['close_option'].min(), 
                     self.data['theoretical_price'].min())
        max_val = max(self.data['close_option'].max(), 
                     self.data['theoretical_price'].max())
        ax3.plot([min_val, max_val], [min_val, max_val], 'r--')
        ax3.set_title('Actual vs Theoretical Prices')
        ax3.set_xlabel('Actual Price')
        ax3.set_ylabel('Theoretical Price')
        ax3.grid(True)
        
        # 4. Error vs Strike Price
        ax4 = fig.add_subplot(324)
        ax4.scatter(self.data['strike_price'], 
                   abs(self.data['theoretical_price'] - self.data['close_option']))
        ax4.set_title('Absolute Error vs Strike Price')
        ax4.set_xlabel('Strike Price')
        ax4.set_ylabel('Absolute Error')
        ax4.grid(True)
        
        # 5. Error vs Time to Expiry
        ax5 = fig.add_subplot(325)
        ax5.scatter(self.data['time_to_expiry'], 
                   abs(self.data['theoretical_price'] - self.data['close_option']))
        ax5.set_title('Absolute Error vs Time to Expiry')
        ax5.set_xlabel('Time to Expiry')
        ax5.set_ylabel('Absolute Error')
        ax5.grid(True)
        
        # 6. Rolling MSE
        ax6 = fig.add_subplot(326)
        squared_error = (self.data['theoretical_price'] - self.data['close_option'])**2
        rolling_mse = squared_error.rolling(window=20).mean()
        ax6.plot(self.data['datetime'], rolling_mse)
        ax6.set_title('20-Period Rolling MSE')
        ax6.set_xlabel('Date')
        ax6.set_ylabel('MSE')
        ax6.grid(True)
        plt.setp(ax6.xaxis.get_ticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.show()


data = pd.read_csv('/home/jjbigdub/gitrepo/quant-insider/call_data.csv')

analyzer = OptionAnalysis(data[:100])

mse_results = analyzer.calculate_mse()
print(f"Mean Squared Error: {mse_results['mse']:.2f}")
print(f"Root Mean Squared Error: {mse_results['rmse']:.2f}")

analyzer.plot_all_analysis()

