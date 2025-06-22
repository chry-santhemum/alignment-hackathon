import plotly.express as px
import pandas as pd


def plot_sycophancy_scores():
    """Plot average normalized scores by sycophancy level using plotly express."""
    
    # Data
    data = {
        'Sycophancy Level': ['strongly_agree', 'weakly_agree', 'neutral', 
                             'weakly_disagree', 'strongly_disagree'],
        'Average Normalized Score': [-0.973, -0.207, 0.000, -0.119, -0.671]
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Define the order for x-axis (from most agreeable to most disagreeable)
    category_order = ['strongly_agree', 'weakly_agree', 'neutral', 
                      'weakly_disagree', 'strongly_disagree']
    
    # Create the line plot
    fig = px.line(
        df, 
        x='Sycophancy Level', 
        y='Average Normalized Score',
        title='Avg reward vs. Sycophancy level',
        category_orders={'Sycophancy Level': category_order},
        markers=True
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title='Sycophancy Level',
        yaxis_title='Average Normalized Score (relative to neutral)',
        xaxis_tickangle=-45,
        height=600,
        width=800,
        font=dict(size=12),
        showlegend=False
    )
    
    # Update trace to show markers and values
    fig.update_traces(
        mode='lines+markers+text',
        marker=dict(size=10),
        text=[f'{val:.3f}' for val in data['Average Normalized Score']],
        textposition='top center',
        line=dict(width=2)
    )
    
    # Adjust y-axis range to accommodate text
    fig.update_yaxes(range=[-1.2, 0.3])
    
    return fig


if __name__ == "__main__":
    # Create and show the plot
    fig = plot_sycophancy_scores()
    # fig.show()
    
    # Optionally save the plot
    # fig.write_html("sycophancy_scores_plot.html")
    
    # Save in high resolution
    fig.write_image("sycophancy_scores_plot.png", 
                    width=800,      # Width in pixels
                    height=600,     # Height in pixels
                    scale=3)         # Scale factor (2 = 2x resolution)
