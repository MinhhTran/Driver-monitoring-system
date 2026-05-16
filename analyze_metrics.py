import pandas as pd

def calculate_metrics(df):
    """Calculates accuracy, true positive rate, and false positive rate."""
    df['System_Alert'] = df['System_Alert'].astype(str).str.lower() == 'true'
    df['Ground_Truth_Fatigue'] = df['Ground_Truth_Fatigue'].astype(str).str.lower() == 'true'

    # Calculate Confusion Matrix elements
    TP = len(df[(df['System_Alert'] == True) & (df['Ground_Truth_Fatigue'] == True)])
    TN = len(df[(df['System_Alert'] == False) & (df['Ground_Truth_Fatigue'] == False)])
    FP = len(df[(df['System_Alert'] == True) & (df['Ground_Truth_Fatigue'] == False)])
    FN = len(df[(df['System_Alert'] == False) & (df['Ground_Truth_Fatigue'] == True)])

    total = TP + TN + FP + FN
    
    # Calculate percentages, handling division by zero
    accuracy = (TP + TN) / total if total > 0 else 0
    tpr = TP / (TP + FN) if (TP + FN) > 0 else 0 # Sensitivity
    fpr = FP / (FP + TN) if (FP + TN) > 0 else 0 # False Positive Rate

    return pd.Series({
        'Samples': total,
        'Accuracy (%)': accuracy * 100,
        'TPR (%)': tpr * 100,
        'FPR (%)': fpr * 100,
        'Avg Latency (ms)': df['Latency_ms'].mean(),
        'Avg RAM (KB)': df['Peak_RAM_KB'].mean()
    })

# Load the logged data
try:
    data = pd.read_csv('dms_performance_log(2).csv')
    print("\n" + "="*50)
    print(" DMS PIPELINE PERFORMANCE ANALYSIS ")
    print("="*50)

    # Group the data by testing Condition and Pipeline, then apply the metrics calculation
    results = data.groupby(['Condition', 'Pipeline']).apply(calculate_metrics).round(2)
    
    # Print the beautifully formatted table directly to the terminal
    print(results.to_string())
    print("="*50 + "\n")
    results.to_csv('dms_final_analysis(2).csv')
    print("Analysis saved to 'dms_final_analysis(2).csv'")

except FileNotFoundError:
    print("Error: 'dms_performance_log.csv' not found. Please run the Camera script first.")