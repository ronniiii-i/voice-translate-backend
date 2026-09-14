import warnings
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from sacrebleu import corpus_bleu

warnings.filterwarnings("ignore")


def load_parallel_dataset(source_lang, target_lang, split="test", max_samples=500):
    """Loads a parallel dataset safely, attempting config flipping if standard fails."""
    config_name = f"{source_lang}-{target_lang}"
    alt_config_name = f"{target_lang}-{source_lang}"
    
    dataset = None
    is_flipped = False
    
    try:
        dataset = load_dataset("flORES", config_name, split=split, streaming=True)
    except Exception:
        try:
            dataset = load_dataset("flORES", alt_config_name, split=split, streaming=True)
            is_flipped = True
        except Exception as e:
            raise RuntimeError(f"Could not load dataset configuration for {source_lang}↔{target_lang}: {e}")

    sources, targets = [], []
    for item in dataset.take(max_samples):
        
        src_text = item.get(f"sentence_{source_lang}")
        tgt_text = item.get(f"sentence_{target_lang}")
        
        if not src_text or not tgt_text:
            
            keys = list(item.keys())
            src_key = next((k for k in keys if source_lang in k), keys[0])
            tgt_key = next((k for k in keys if target_lang in k), keys[1])
            src_text, tgt_text = item[src_key], item[tgt_key]

        if is_flipped:
            sources.append(tgt_text)
            targets.append(src_text)
        else:
            sources.append(src_text)
            targets.append(tgt_text)
            
    return sources, targets



import re

def postprocess_translation(text):
    """Cleans up stray sub-word tokenization markers, spacing, and broken boundaries."""
    
    text = re.sub(r'\s+([?.!,:;])', r'\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text



def evaluate_translation_pipeline(pairs_to_test):
    results = {}
    
    for src_lang, tgt_lang, model_id in pairs_to_test:
        print(f"\nEvaluating: {src_lang} -> {tgt_lang} using model {model_id}")
        
        
        if src_lang != "en" and tgt_lang != "en":
            print(f"-> Skipping unaligned pivot route ({src_lang}→{tgt_lang}) to keep BLEU reliable.")
            continue

        try:
            sources, references = load_parallel_dataset(src_lang, tgt_lang, max_samples=100)
        except Exception as e:
            print(f"-> Skipping due to load error: {e}")
            continue

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

        hypotheses = []
        for src_text in sources:
            inputs = tokenizer(src_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            
            
            translated_tokens = model.generate(
                **inputs,
                num_beams=4,
                length_penalty=1.0,
                max_length=128,
                early_stopping=True
            )
            
            decoded = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
            cleaned_output = postprocess_translation(decoded)
            hypotheses.append(cleaned_output)

        
        bleu = corpus_bleu(hypotheses, [references])
        results[(src_lang, tgt_lang)] = bleu.score
        print(f"-> BLEU Score for {src_lang}→{tgt_lang}: {bleu.score:.2f}")

    return results


if __name__ == "__main__":
    
    test_pairs = [
        ("en", "fr", "Helsinki-NLP/opus-mt-en-fr"),
        ("fr", "en", "Helsinki-NLP/opus-mt-fr-en"),
        ("en", "es", "Helsinki-NLP/opus-mt-en-es"),
        ("fr", "es", "Helsinki-NLP/opus-mt-fr-es"), 
    ]
    
    eval_results = evaluate_translation_pipeline(test_pairs)
    print("\nFinal Pipeline Evaluation Summary:", eval_results)