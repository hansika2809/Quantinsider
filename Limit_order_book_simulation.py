import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import RMSprop
import matplotlib.pyplot as plt
from collections import defaultdict
import seaborn as sns
from typing import Tuple, Dict

class CTLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super(CTLSTM, self).__init__()
        self.hidden_size = hidden_size

        self.input_gate = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.Tanh()
        )
        self.forget_gate = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.Tanh()
        )
        self.output_gate = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.Tanh()
        )
        self.cell_gate = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.Tanh()
        )
        self.decay_gate = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.Softplus()
        )

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor, c_prev: torch.Tensor, delta_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        
        combined = torch.cat([x, h_prev], dim=1)
        i = self.input_gate(combined)
        f = self.forget_gate(combined)
        o = self.output_gate(combined)
        c_tilde = self.cell_gate(combined)
        decay = self.decay_gate(combined)
        c_decay = c_prev * torch.exp(-decay * delta_t.unsqueeze(1))
        c_next = f * c_decay + i * c_tilde
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

class NeuralHawkesProcess(nn.Module):
    def __init__(self, num_events: int, hidden_size: int):
        super(NeuralHawkesProcess, self).__init__()
        self.num_events = num_events
        self.hidden_size = hidden_size

        self.event_embedding = nn.Embedding(num_events, hidden_size)
        self.ct_lstm = CTLSTM(hidden_size, hidden_size)
        self.intensity_layer = nn.Sequential(
            nn.Linear(hidden_size, num_events),
            nn.Softplus()
        )

    def forward(self, events: torch.Tensor, times: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = events.size(0)
        seq_len = events.size(1)
        
        h = torch.zeros(batch_size, self.hidden_size).to(events.device)
        c = torch.zeros(batch_size, self.hidden_size).to(events.device)
        
        intensities = []
        hidden_states = []
        
        for t in range(seq_len):
          
            event_embed = self.event_embedding(events[:, t])
            delta_t = times[:, t] if t == 0 else times[:, t] - times[:, t-1]
            h, c = self.ct_lstm(event_embed, h, c, delta_t)
            intensity = self.intensity_layer(h)
            intensities.append(intensity)
            hidden_states.append(h)
            
        return torch.stack(intensities, dim=1), torch.stack(hidden_states, dim=1)

class LOBSimulator:
    def __init__(self, model: NeuralHawkesProcess, price_levels: int = 5):
        self.model = model
        self.price_levels = price_levels
        self.current_state = defaultdict(lambda: defaultdict(float))
        self.mid_price_history = []
        self.volume_history = []
        self.time_history = []
        
        self.fig = None
        self.axes = None
        self.orderbook_ax = None
        self.price_ax = None
        self.animation = None
        
    def initialize_orderbook(self):
        base_price = 100.0
        for i in range(self.price_levels):
            self.current_state['ask'][base_price + i] = 100
            self.current_state['bid'][base_price - i - 1] = 100
            
    def simulate_step(self, current_time: float, market_condition: str = 'neutral') -> Dict:
        with torch.no_grad():
            h = torch.zeros(1, self.model.hidden_size)
            c = torch.zeros(1, self.model.hidden_size)
            intensity = self.model.intensity_layer(h)
            
            event_probs = intensity.softmax(dim=-1)
            event_type = torch.multinomial(event_probs, 1).item()
            
            # adjust probabilities based on market condition
            if market_condition == 'buy':
                event_probs[:, 0] *= 2  # increase buy probability
            elif market_condition == 'sell':
                event_probs[:, 1] *= 2  # increase sell probability
        
        price = self._generate_price(event_type)
        volume = self._generate_volume()
        self._update_orderbook(event_type, price, volume)
        mid_price = self.get_mid_price()
        self.mid_price_history.append(mid_price)
        self.volume_history.append(volume)
        self.time_history.append(current_time)
        
        return {
            'event_type': event_type,
            'price': price,
            'volume': volume,
            'mid_price': mid_price
        }
    
    def _generate_price(self, event_type: int) -> float:
        # generate price following power law distribution
        current_mid = self.get_mid_price()
        alpha = 1.5 if event_type == 0 else 4.3  # different power law coefficients for bid/ask
        price_offset = np.random.power(alpha)
        return current_mid + price_offset if event_type == 0 else current_mid - price_offset
    
    def _generate_volume(self) -> int:
        # generate volume following power law distribution
        return int(np.random.power(1.5) * 100)
    
    def get_mid_price(self) -> float:
        best_bid = max(self.current_state['bid'].keys()) if self.current_state['bid'] else 0
        best_ask = min(self.current_state['ask'].keys()) if self.current_state['ask'] else 0
        return (best_bid + best_ask) / 2
    
    def _update_orderbook(self, event_type: int, price: float, volume: int):
        if event_type == 0: 
            self.current_state['bid'][price] += volume
        else: 
            self.current_state['ask'][price] += volume
            
    def plot_results(self):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # plot mid-price evolution
        ax1.plot(self.time_history, self.mid_price_history)
        ax1.set_title('Mid-price Evolution')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Price')
        
        # plot volatility clustering
        returns = np.diff(np.log(self.mid_price_history))
        squared_returns = returns**2
        lags = range(1, 21)
        autocorr = [np.corrcoef(squared_returns[:-lag], squared_returns[lag:])[0,1] for lag in lags]
        ax2.plot(lags, autocorr, 'g-', label='Simulated')
        ax2.set_title('Volatility Clustering')
        ax2.set_xlabel('Lag')
        ax2.set_ylabel('Autocorrelation')
        
        # plot log returns distribution
        sns.histplot(returns, kde=True, ax=ax3)
        ax3.set_title('Log Returns Distribution')
        
        # plot inter-arrival times
        inter_arrival_times = np.diff(self.time_history)
        sns.histplot(inter_arrival_times, kde=True, ax=ax4)
        ax4.set_title('Inter-arrival Times Distribution')
        
        plt.tight_layout()
        plt.show()

def train_model(model: NeuralHawkesProcess, train_data: Tuple[torch.Tensor, torch.Tensor], 
                num_epochs: int = 200, learning_rate: float = 2e-3):
    optimizer = RMSprop(model.parameters(), lr=learning_rate)
    events, times = train_data
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        intensities, _ = model(events, times)
        loss = compute_hawkes_loss(intensities, events, times)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

def compute_hawkes_loss(intensities: torch.Tensor, events: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
    # compute negative log likelihood loss for Hawkes process
    log_likelihood = 0
    batch_size = events.size(0)
    for b in range(batch_size):
        for t in range(events.size(1)):
            event_type = events[b, t]
            log_likelihood += torch.log(intensities[b, t, event_type] + 1e-8)
    
    total_time = times[:, -1] - times[:, 0]
    intensity_integral = intensities.sum(dim=(1, 2)) * total_time
    log_likelihood -= intensity_integral.sum()
    
    return -log_likelihood

#-----------------------------------------------------------------------------------------------------------------------

num_events = 2  # bid and ask events
hidden_size = 16

model = NeuralHawkesProcess(num_events, hidden_size)

# generate synthetic training data
num_sequences = 100
seq_length = 50
events = torch.randint(0, num_events, (num_sequences, seq_length))
times = torch.cumsum(torch.exp(torch.randn(num_sequences, seq_length)), dim=1)

train_model(model, (events, times))

simulator = LOBSimulator(model)
simulator.initialize_orderbook()

# simulate orderbook
num_steps = 1000
current_time = 0
market_conditions = ['buy'] * 333 + ['neutral'] * 334 + ['sell'] * 333

for i in range(num_steps):
    current_time += np.random.exponential(1/100)  # Average 100 events per unit time
    simulator.simulate_step(current_time, market_conditions[i // 3])
simulator.plot_results()

