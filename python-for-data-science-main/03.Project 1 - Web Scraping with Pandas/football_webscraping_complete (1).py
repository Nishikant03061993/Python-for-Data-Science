"""
Football Data Web Scraping Project - Complete Visualization Suite
================================================================

This project demonstrates how to scrape football match data from the web using Python and Pandas,
with comprehensive visualization using ALL major Python visualization libraries.

Author: [Your Name]
Date: December 2024
Purpose: Educational demonstration of web scraping and data visualization techniques
"""

# ============================================================================
# SECTION 1: IMPORTING ALL VISUALIZATION LIBRARIES
# ============================================================================

# Core data manipulation and analysis
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Basic visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Interactive visualization libraries
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.figure_factory as ff
    PLOTLY_AVAILABLE = True
except ImportError:
    print("⚠️ Plotly not installed. Install with: pip install plotly")
    PLOTLY_AVAILABLE = False

# Statistical analysis
try:
    from scipy import stats
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    SCIPY_SKLEARN_AVAILABLE = True
except ImportError:
    print("⚠️ Scipy/Sklearn not installed. Install with: pip install scipy scikit-learn")
    SCIPY_SKLEARN_AVAILABLE = False

# Specialized visualization tools
try:
    import squarify  # For treemaps
    SQUARIFY_AVAILABLE = True
except ImportError:
    print("⚠️ squarify not installed. Install with: pip install squarify")
    SQUARIFY_AVAILABLE = False

try:
    import networkx as nx  # For network graphs
    NETWORKX_AVAILABLE = True
except ImportError:
    print("⚠️ NetworkX not installed. Install with: pip install networkx")
    NETWORKX_AVAILABLE = False

try:
    from wordcloud import WordCloud  # For word clouds
    WORDCLOUD_AVAILABLE = True
except ImportError:
    print("⚠️ WordCloud not installed. Install with: pip install wordcloud")
    WORDCLOUD_AVAILABLE = False

# Set visualization styles
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
if PLOTLY_AVAILABLE:
    px.defaults.template = "plotly_white"

print("📚 All available visualization libraries imported successfully!")
print("=" * 70)

# ============================================================================
# SECTION 2: DATA LOADING FUNCTIONS
# ============================================================================

def load_single_league_data(season='2526', league='E0'):
    """Load data for a single league and season."""
    url = f'https://www.football-data.co.uk/mmz4281/{season}/{league}.csv'
    print(f"🔄 Loading data from: {url}")
    
    try:
        df = pd.read_csv(url)
        print(f"✅ Successfully loaded {len(df)} matches")
        return df
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

def load_multiple_leagues(season='2122'):
    """Load data for multiple English football leagues."""
    print(f"\n🔄 LOADING MULTIPLE LEAGUES DATA (Season {season})")
    print("=" * 60)
    
    leagues_config = {
        'Premier League': {'code': 'E0', 'tier': 1},
        'Championship': {'code': 'E1', 'tier': 2}, 
        'League One': {'code': 'E2', 'tier': 3},
        'League Two': {'code': 'E3', 'tier': 4},
        'Conference': {'code': 'EC', 'tier': 5}
    }
    
    leagues_data = {}
    
    for league_name, config in leagues_config.items():
        url = f'https://www.football-data.co.uk/mmz4281/{season}/{config["code"]}.csv'
        
        try:
            print(f"📥 Loading {league_name} ({config['code']})...")
            df = pd.read_csv(url)
            df['League'] = league_name
            df['Tier'] = config['tier']
            df['League_Code'] = config['code']
            leagues_data[league_name] = df
            print(f"   ✅ Success: {len(df)} matches loaded")
        except Exception as e:
            print(f"   ❌ Failed to load {league_name}: {e}")
    
    print(f"\n📊 SUMMARY: {len(leagues_data)} leagues loaded successfully")
    return leagues_data

# ============================================================================
# SECTION 3: MATPLOTLIB VISUALIZATIONS
# ============================================================================

def create_matplotlib_visualizations(leagues_data):
    """Create comprehensive matplotlib visualizations."""
    print(f"\n📊 CREATING MATPLOTLIB VISUALIZATIONS")
    print("=" * 50)
    
    # Figure 1: Multi-panel league comparison
    fig, axes = plt.subplots(3, 3, figsize=(20, 18))
    fig.suptitle('English Football Leagues Analysis - Matplotlib Suite', fontsize=20, fontweight='bold')
    
    # Prepare data
    league_names = []
    avg_goals = []
    home_win_pct = []
    total_matches = []
    
    for league_name, df in leagues_data.items():
        if 'FTHG' in df.columns and 'FTAG' in df.columns:
            league_names.append(league_name)
            total_goals = df['FTHG'] + df['FTAG']
            avg_goals.append(total_goals.mean())
            home_win_pct.append((df['FTR'] == 'H').sum() / len(df) * 100)
            total_matches.append(len(df))
    
    # Plot 1: Bar chart with annotations
    bars1 = axes[0, 0].bar(league_names, avg_goals, color='skyblue', alpha=0.8, edgecolor='navy')
    axes[0, 0].set_title('Average Goals per Match', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('Average Goals')
    axes[0, 0].tick_params(axis='x', rotation=45)
    for bar, val in zip(bars1, avg_goals):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                       f'{val:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Horizontal bar chart
    axes[0, 1].barh(league_names, home_win_pct, color='lightgreen', alpha=0.8)
    axes[0, 1].set_title('Home Win Percentage', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Home Win %')
    
    # Plot 3: Stacked histogram
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for i, (league_name, df) in enumerate(leagues_data.items()):
        if 'FTHG' in df.columns and 'FTAG' in df.columns:
            total_goals = df['FTHG'] + df['FTAG']
            axes[0, 2].hist(total_goals, bins=range(0, 12), alpha=0.6, 
                          label=league_name, color=colors[i % len(colors)])
    axes[0, 2].set_title('Goals Distribution')
    axes[0, 2].legend()
    
    # Plot 4: Pie chart with explosion
    if leagues_data:
        first_league = list(leagues_data.values())[0]
        if 'FTR' in first_league.columns:
            result_counts = first_league['FTR'].value_counts()
            explode = (0.05, 0, 0.05)
            axes[1, 0].pie(result_counts.values, labels=['Home Win', 'Draw', 'Away Win'], 
                          autopct='%1.1f%%', explode=explode, shadow=True)
            axes[1, 0].set_title('Match Results Distribution')
    
    # Plot 5: Box plot
    goal_data = []
    for league_name, df in leagues_data.items():
        if 'FTHG' in df.columns and 'FTAG' in df.columns:
            total_goals = df['FTHG'] + df['FTAG']
            goal_data.append(total_goals)
    
    bp = axes[1, 1].boxplot(goal_data, labels=league_names, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[1, 1].set_title('Goals Box Plot')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    # Plot 6: Scatter plot with color mapping
    if leagues_data:
        first_league = list(leagues_data.values())[0]
        if 'FTHG' in first_league.columns and 'FTAG' in first_league.columns:
            scatter = axes[1, 2].scatter(first_league['FTHG'], first_league['FTAG'], 
                                       c=first_league['FTHG'] + first_league['FTAG'], 
                                       cmap='viridis', alpha=0.6, s=50)
            axes[1, 2].set_title('Home vs Away Goals')
            axes[1, 2].set_xlabel('Home Goals')
            axes[1, 2].set_ylabel('Away Goals')
            plt.colorbar(scatter, ax=axes[1, 2])
    
    # Plot 7: Area chart
    if len(league_names) > 0:
        x = np.arange(len(league_names))
        axes[2, 0].fill_between(x, avg_goals, alpha=0.7, color='lightblue')
        axes[2, 0].plot(x, avg_goals, marker='o', linewidth=2, markersize=8)
        axes[2, 0].set_xticks(x)
        axes[2, 0].set_xticklabels(league_names, rotation=45)
        axes[2, 0].set_title('Goals Trend Area Chart')
    
    # Plot 8: Polar plot
    angles = np.linspace(0, 2*np.pi, len(league_names), endpoint=False)
    axes[2, 1] = plt.subplot(3, 3, 8, projection='polar')
    axes[2, 1].plot(angles, avg_goals, 'o-', linewidth=2, markersize=8)
    axes[2, 1].fill(angles, avg_goals, alpha=0.25)
    axes[2, 1].set_xticks(angles)
    axes[2, 1].set_xticklabels(league_names)
    axes[2, 1].set_title('Goals Radar Chart')
    
    # Plot 9: Stem plot
    axes[2, 2].stem(league_names, home_win_pct, basefmt=" ")
    axes[2, 2].set_title('Home Win % Stem Plot')
    axes[2, 2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    print("✅ Matplotlib visualizations completed!")

# ============================================================================
# SECTION 4: SEABORN VISUALIZATIONS
# ============================================================================

def create_seaborn_visualizations(leagues_data):
    """Create advanced seaborn visualizations."""
    print(f"\n🎨 CREATING SEABORN VISUALIZATIONS")
    print("=" * 50)
    
    combined_df = pd.concat([df for df in leagues_data.values()], ignore_index=True)
    combined_df['Total_Goals'] = combined_df['FTHG'] + combined_df['FTAG']
    
    # Figure 1: Statistical plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Seaborn Statistical Visualizations', fontsize=16, fontweight='bold')
    
    # Correlation heatmap
    numeric_cols = combined_df.select_dtypes(include=[np.number]).columns
    correlation_matrix = combined_df[numeric_cols].corr()
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                center=0, ax=axes[0, 0], fmt='.2f')
    axes[0, 0].set_title('Correlation Heatmap')
    
    # Violin plot
    sns.violinplot(data=combined_df, x='League', y='Total_Goals', ax=axes[0, 1])
    axes[0, 1].set_title('Goals Distribution by League')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Pair plot (subset)
    if len(numeric_cols) > 3:
        subset_cols = ['FTHG', 'FTAG', 'Total_Goals']
        sns.scatterplot(data=combined_df, x='FTHG', y='FTAG', hue='League', ax=axes[0, 2])
        axes[0, 2].set_title('Home vs Away Goals')
    
    # Box plot with swarm
    sns.boxplot(data=combined_df, x='League', y='Total_Goals', ax=axes[1, 0])
    sns.swarmplot(data=combined_df, x='League', y='Total_Goals', ax=axes[1, 0], 
                  size=3, alpha=0.5)
    axes[1, 0].set_title('Goals Box + Swarm Plot')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Count plot
    sns.countplot(data=combined_df, x='FTR', hue='League', ax=axes[1, 1])
    axes[1, 1].set_title('Result Counts by League')
    
    # Regression plot
    if 'FTHG' in combined_df.columns and 'FTAG' in combined_df.columns:
        sns.regplot(data=combined_df, x='FTHG', y='FTAG', ax=axes[1, 2], scatter_kws={'alpha':0.5})
        axes[1, 2].set_title('Home vs Away Goals Regression')
    
    plt.tight_layout()
    plt.show()
    
    # Figure 2: Distribution plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Seaborn Distribution Analysis', fontsize=16, fontweight='bold')
    
    # Distribution plot
    for league, df in leagues_data.items():
        if 'FTHG' in df.columns and 'FTAG' in df.columns:
            total_goals = df['FTHG'] + df['FTAG']
            sns.histplot(total_goals, kde=True, alpha=0.6, label=league, ax=axes[0, 0])
    axes[0, 0].set_title('Goals Distribution with KDE')
    axes[0, 0].legend()
    
    # Joint plot equivalent
    if 'FTHG' in combined_df.columns and 'FTAG' in combined_df.columns:
        sns.scatterplot(data=combined_df, x='FTHG', y='FTAG', hue='League', ax=axes[0, 1])
        axes[0, 1].set_title('Joint Distribution')
    
    # Categorical plot
    sns.barplot(data=combined_df, x='League', y='Total_Goals', ax=axes[1, 0])
    axes[1, 0].set_title('Average Goals by League')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Strip plot
    sns.stripplot(data=combined_df, x='League', y='Total_Goals', ax=axes[1, 1], 
                  size=4, alpha=0.7)
    axes[1, 1].set_title('Goals Strip Plot')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    print("✅ Seaborn visualizations completed!")

# ============================================================================
# SECTION 5: PLOTLY INTERACTIVE VISUALIZATIONS
# ============================================================================

def create_plotly_visualizations(leagues_data):
    """Create interactive Plotly visualizations."""
    if not PLOTLY_AVAILABLE:
        print("❌ Plotly not available - skipping interactive visualizations")
        return
    
    print(f"\n🚀 CREATING INTERACTIVE PLOTLY VISUALIZATIONS")
    print("=" * 50)
    
    combined_df = pd.concat([df for df in leagues_data.values()], ignore_index=True)
    combined_df['Total_Goals'] = combined_df['FTHG'] + combined_df['FTAG']
    
    # 1. Interactive scatter plot
    fig1 = px.scatter(combined_df, x='FTHG', y='FTAG', color='League', 
                     size='Total_Goals', hover_data=['HomeTeam', 'AwayTeam'],
                     title='Interactive Home vs Away Goals Analysis',
                     labels={'FTHG': 'Home Team Goals', 'FTAG': 'Away Team Goals'})
    fig1.show()
    
    # 2. Box plot comparison
    fig2 = px.box(combined_df, x='League', y='Total_Goals', 
                  title='Goals Distribution Across Leagues',
                  color='League')
    fig2.show()
    
    # 3. Sunburst chart
    result_summary = combined_df.groupby(['League', 'FTR']).size().reset_index(name='Count')
    fig3 = px.sunburst(result_summary, path=['League', 'FTR'], values='Count',
                       title='Match Results Distribution by League')
    fig3.show()
    
    # 4. 3D scatter plot
    fig4 = px.scatter_3d(combined_df, x='FTHG', y='FTAG', z='Total_Goals',
                        color='League', title='3D Goals Analysis',
                        labels={'FTHG': 'Home Goals', 'FTAG': 'Away Goals'})
    fig4.show()
    
    # 5. Animated bar chart (if date available)
    try:
        combined_df['Date'] = pd.to_datetime(combined_df['Date'], format='%d/%m/%Y')
        combined_df['Month'] = combined_df['Date'].dt.month
        monthly_data = combined_df.groupby(['League', 'Month'])['Total_Goals'].mean().reset_index()
        
        fig5 = px.bar(monthly_data, x='League', y='Total_Goals', 
                     animation_frame='Month', color='League',
                     title='Monthly Goals Animation')
        fig5.show()
    except:
        print("Date animation skipped - date parsing failed")
    
    # 6. Treemap
    league_summary = combined_df.groupby('League').agg({
        'Total_Goals': 'sum',
        'FTHG': 'count'
    }).reset_index()
    league_summary.columns = ['League', 'Total_Goals', 'Matches']
    
    fig6 = px.treemap(league_summary, path=['League'], values='Matches',
                     color='Total_Goals', title='League Treemap')
    fig6.show()
    
    # 7. Parallel coordinates
    numeric_data = combined_df.select_dtypes(include=[np.number]).head(100)
    fig7 = px.parallel_coordinates(numeric_data, color='Total_Goals',
                                  title='Parallel Coordinates Plot')
    fig7.show()
    
    print("✅ Interactive Plotly visualizations completed!")

# ============================================================================
# SECTION 6: ADVANCED STATISTICAL VISUALIZATIONS
# ============================================================================

def create_advanced_visualizations(leagues_data):
    """Create advanced statistical visualizations."""
    if not SCIPY_SKLEARN_AVAILABLE:
        print("❌ Scipy/Sklearn not available - skipping advanced visualizations")
        return
    
    print(f"\n🔬 CREATING ADVANCED STATISTICAL VISUALIZATIONS")
    print("=" * 50)
    
    combined_df = pd.concat([df for df in leagues_data.values()], ignore_index=True)
    combined_df['Total_Goals'] = combined_df['FTHG'] + combined_df['FTAG']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Advanced Statistical Analysis', fontsize=16, fontweight='bold')
    
    # 1. Q-Q Plot
    stats.probplot(combined_df['Total_Goals'], dist="norm", plot=axes[0, 0])
    axes[0, 0].set_title('Q-Q Plot: Goals vs Normal Distribution')
    
    # 2. Distribution fitting
    axes[0, 1].hist(combined_df['Total_Goals'], bins=20, density=True, alpha=0.7)
    x = np.linspace(combined_df['Total_Goals'].min(), combined_df['Total_Goals'].max(), 100)
    axes[0, 1].plot(x, stats.norm.pdf(x, combined_df['Total_Goals'].mean(), 
                                     combined_df['Total_Goals'].std()), 'r-', label='Normal')
    axes[0, 1].set_title('Distribution Fitting')
    axes[0, 1].legend()
    
    # 3. PCA Analysis
    numeric_cols = combined_df.select_dtypes(include=[np.number]).columns
    numeric_data = combined_df[numeric_cols].fillna(0)
    
    if len(numeric_data.columns) > 2:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_data)
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(scaled_data)
        
        scatter = axes[0, 2].scatter(pca_result[:, 0], pca_result[:, 1], 
                                   c=combined_df['Total_Goals'], cmap='viridis', alpha=0.6)
        axes[0, 2].set_title(f'PCA Analysis')
        axes[0, 2].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2f})')
        axes[0, 2].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2f})')
        plt.colorbar(scatter, ax=axes[0, 2])
    
    # 4. K-Means Clustering
    if len(numeric_data.columns) > 2:
        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(scaled_data)
        scatter2 = axes[1, 0].scatter(pca_result[:, 0], pca_result[:, 1], 
                                    c=clusters, cmap='Set1', alpha=0.6)
        axes[1, 0].set_title('K-Means Clustering')
    
    # 5. Residual plot
    if 'FTHG' in combined_df.columns and 'FTAG' in combined_df.columns:
        from sklearn.linear_model import LinearRegression
        X = combined_df[['FTHG']].values
        y = combined_df['FTAG'].values
        
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        residuals = y - y_pred
        
        axes[1, 1].scatter(y_pred, residuals, alpha=0.6)
        axes[1, 1].axhline(y=0, color='red', linestyle='--')
        axes[1, 1].set_title('Residual Plot')
        axes[1, 1].set_xlabel('Predicted Values')
        axes[1, 1].set_ylabel('Residuals')
    
    # 6. Confidence intervals
    league_goals = []
    league_names = []
    for league, df in leagues_data.items():
        goals = df['FTHG'] + df['FTAG']
        league_goals.append(goals)
        league_names.append(league)
    
    means = [np.mean(goals) for goals in league_goals]
    stds = [np.std(goals) for goals in league_goals]
    
    axes[1, 2].errorbar(range(len(league_names)), means, yerr=stds, 
                       fmt='o', capsize=5, capthick=2)
    axes[1, 2].set_xticks(range(len(league_names)))
    axes[1, 2].set_xticklabels(league_names, rotation=45)
    axes[1, 2].set_title('Goals with Confidence Intervals')
    
    plt.tight_layout()
    plt.show()
    print("✅ Advanced statistical visualizations completed!")

# ============================================================================
# SECTION 7: SPECIALIZED VISUALIZATIONS
# ============================================================================

def create_specialized_visualizations(leagues_data):
    """Create specialized visualizations using additional libraries."""
    print(f"\n🎭 CREATING SPECIALIZED VISUALIZATIONS")
    print("=" * 50)
    
    # 1. Network Graph
    if NETWORKX_AVAILABLE and 'Premier League' in leagues_data:
        try:
            df_pl = leagues_data['Premier League']
            G = nx.Graph()
            
            for _, match in df_pl.iterrows():
                home_team = match['HomeTeam']
                away_team = match['AwayTeam']
                total_goals = match['FTHG'] + match['FTAG']
                
                if G.has_edge(home_team, away_team):
                    G[home_team][away_team]['weight'] += total_goals
                else:
                    G.add_edge(home_team, away_team, weight=total_goals)
            
            plt.figure(figsize=(15, 15))
            pos = nx.spring_layout(G, k=3, iterations=50)
            
            edges = G.edges()
            weights = [G[u][v]['weight']/10 for u, v in edges]
            nx.draw_networkx_edges(G, pos, width=weights, alpha=0.6)
            nx.draw_networkx_nodes(G, pos, node_size=1000, node_color='lightblue', alpha=0.8)
            nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
            
            plt.title('Team Interaction Network', fontsize=16, fontweight='bold')
            plt.axis('off')
            plt.show()
            print("✅ Network graph completed!")
        except Exception as e:
            print(f"Network graph failed: {e}")
    
    # 2. Treemap
    if SQUARIFY_AVAILABLE:
        try:
            league_sizes = [len(df) for df in leagues_data.values()]
            league_names = list(leagues_data.keys())
            
            plt.figure(figsize=(12, 8))
            colors = plt.cm.Set3(np.linspace(0, 1, len(league_names)))
            
            squarify.plot(sizes=league_sizes, label=league_names, color=colors, alpha=0.8)
            plt.title('League Sizes Treemap', fontsize=16, fontweight='bold')
            plt.axis('off')
            plt.show()
            print("✅ Treemap completed!")
        except Exception as e:
            print(f"Treemap failed: {e}")
    
    # 3. Word Cloud (team names)
    if WORDCLOUD_AVAILABLE and leagues_data:
        try:
            all_teams = []
            for df in leagues_data.values():
                all_teams.extend(df['HomeTeam'].tolist())
                all_teams.extend(df['AwayTeam'].tolist())
            
            team_text = ' '.join(all_teams)
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(team_text)
            
            plt.figure(figsize=(12, 6))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title('Team Names Word Cloud', fontsize=16, fontweight='bold')
            plt.show()
            print("✅ Word cloud completed!")
        except Exception as e:
            print(f"Word cloud failed: {e}")
    
    # 4. Radar Chart
    try:
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        metrics = ['Avg_Goals', 'Home_Win_Pct', 'Away_Win_Pct', 'Draw_Pct']
        league_metrics = {}
        
        for league, df in leagues_data.items():
            if 'FTHG' in df.columns and 'FTAG' in df.columns:
                total_goals = df['FTHG'] + df['FTAG']
                results = df['FTR'].value_counts(normalize=True) * 100
                
                league_metrics[league] = [
                    total_goals.mean(),
                    results.get('H', 0),
                    results.get('A', 0), 
                    results.get('D', 0)
                ]
        
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]
        
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        for i, (league, values) in enumerate(league_metrics.items()):
            values += values[:1]
            ax.plot(angles, values, 'o-', linewidth=2, label=league, color=colors[i % len(colors)])
            ax.fill(angles, values, alpha=0.25, color=colors[i % len(colors)])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_title('League Performance Radar Chart', size=16, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        plt.show()
        print("✅ Radar chart completed!")
    except Exception as e:
        print(f"Radar chart failed: {e}")

# ============================================================================
# SECTION 8: MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("🏈 FOOTBALL DATA WEB SCRAPING & VISUALIZATION PROJECT")
    print("=" * 60)
    
    # Load data
    leagues_data = load_multiple_leagues(season='2122')
    
    if not leagues_data:
        print("❌ No data loaded - cannot create visualizations")
        return
    
    print(f"\n🎨 CREATING COMPREHENSIVE VISUALIZATION SUITE")
    print("=" * 60)
    
    # Create all visualizations
    try:
        create_matplotlib_visualizations(leagues_data)
    except Exception as e:
        print(f"❌ Matplotlib visualizations failed: {e}")
    
    try:
        create_seaborn_visualizations(leagues_data)
    except Exception as e:
        print(f"❌ Seaborn visualizations failed: {e}")
    
    try:
        create_plotly_visualizations(leagues_data)
    except Exception as e:
        print(f"❌ Plotly visualizations failed: {e}")
    
    try:
        create_advanced_visualizations(leagues_data)
    except Exception as e:
        print(f"❌ Advanced visualizations failed: {e}")
    
    try:
        create_specialized_visualizations(leagues_data)
    except Exception as e:
        print(f"❌ Specialized visualizations failed: {e}")
    
    print(f"\n✨ COMPREHENSIVE VISUALIZATION PROJECT COMPLETED! ✨")
    print("=" * 70)
    print("📊 VISUALIZATION LIBRARIES USED:")
    print("   ✅ Matplotlib - Static plots and charts")
    print("   ✅ Seaborn - Statistical visualizations")
    if PLOTLY_AVAILABLE:
        print("   ✅ Plotly - Interactive visualizations")
    if NETWORKX_AVAILABLE:
        print("   ✅ NetworkX - Network graphs")
    if SQUARIFY_AVAILABLE:
        print("   ✅ Squarify - Treemap visualizations")
    if WORDCLOUD_AVAILABLE:
        print("   ✅ WordCloud - Text visualizations")
    if SCIPY_SKLEARN_AVAILABLE:
        print("   ✅ Scipy/Sklearn - Statistical analysis")
    
    print("\n🎨 VISUALIZATION TYPES CREATED:")
    print("   • Bar charts, histograms, and area charts")
    print("   • Scatter plots, bubble charts, and 3D plots")
    print("   • Box plots, violin plots, and swarm plots")
    print("   • Heatmaps and correlation matrices")
    print("   • Pie charts and sunburst charts")
    print("   • Interactive dashboards and animations")
    print("   • Statistical distribution and Q-Q plots")
    print("   • PCA and clustering visualizations")
    print("   • Radar charts and polar plots")
    print("   • Network graphs and treemaps")
    print("   • Word clouds and specialized plots")
    print("=" * 70)

# Run the main function
if __name__ == "__main__":
    main()