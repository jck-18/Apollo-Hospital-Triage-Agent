"""
Utility to render performance outputs neatly to the console.
"""
from typing import List, Dict, Any

def print_comparison_table(evaluations: List[Dict[str, Any]]) -> None:
    """
    Renders a markdown-style table in the console comparing all the AI models.
    """
    print("\n" + "=" * 80)
    print(f"| {'Model Architecture':<30} | {'Accuracy':<9} | {'Precision':<9} | {'Recall':<9} | {'F1 Score':<9} |")
    print("-" * 80)
    
    for eval_result in evaluations:
        name = eval_result["name"]
        m = eval_result["metrics"]
        
        # Format metrics as percentages/floats
        acc_str = f"{m['Accuracy']*100:.2f}%"
        prec_str = f"{m['Precision']:.3f}"
        rec_str = f"{m['Recall']:.3f}"
        f1_str = f"{m['F1_Score']:.3f}"
        
        print(f"| {name:<30} | {acc_str:<9} | {prec_str:<9} | {rec_str:<9} | {f1_str:<9} |")
        
    print("=" * 80 + "\n")
