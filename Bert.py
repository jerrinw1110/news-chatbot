
from bert_score import score

def evaluate_with_bertscore(generated_summary, reference_summary):
    P, R, F1 = score([generated_summary], [reference_summary], lang='en', verbose=True)
    return {'precision': P.mean().item(), 'recall': R.mean().item(), 'f1': F1.mean().item()}


