"""
Dietary Menu Recommendation System - Data Preprocessing Pipeline
Indonesian Food Composition Table (TKPI) and URT Integration
Version: 3.0 - Category-aware Fuzzy Matching
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
from difflib import SequenceMatcher

# ============================================================================
# FUZZY MATCHING ENGINE
# ============================================================================

def levenshtein_similarity(s1, s2):
    """Calculate Levenshtein similarity score (0-100)."""
    if pd.isna(s1) or pd.isna(s2): return 0.0
    s1, s2 = str(s1).lower().strip(), str(s2).lower().strip()
    if not s1 or not s2: return 0.0
    return round(SequenceMatcher(None, s1, s2).ratio() * 100, 2)
def find_best_urt_match_by_category(tkpi_row, urt_data, similarity_threshold=85):
    """Find best URT match using exact name or category-aware fuzzy matching."""
    tkpi_name, tkpi_category = tkpi_row['food_name_normalized'], tkpi_row['category']
    if pd.isna(tkpi_name) or pd.isna(tkpi_category): return None

    # Step 1: Exact match (With the new sorter, "ayam daging" == "ayam daging")
    exact_matches = urt_data[urt_data['urt_food_name_normalized'] == tkpi_name]
    if not exact_matches.empty:
        match = exact_matches.iloc[0]
        return {'urt_food_name': match['urt_food_name'], 'urt': match['urt'], 
                'gram_per_portion': match['gram_per_portion'], 'similarity_score': 100.0, 'match_type': 'exact_name'}

    # Step 2: Fuzzy match (Now includes URT items that lack a category!)
    same_cat_urt = urt_data[
        ((urt_data['category'] == tkpi_category) | 
         (urt_data['category'].isna()) | 
         (urt_data['category'] == 'unclassified')) & 
        (urt_data['urt_food_name_normalized'].notna())
    ]
    
    if same_cat_urt.empty: return None

    best_match, best_sim = None, 0
    for _, row in same_cat_urt.iterrows():
        sim = levenshtein_similarity(tkpi_name, row['urt_food_name_normalized'])
        if sim > best_sim:
            best_sim, best_match = sim, row

    if best_sim >= similarity_threshold and best_match is not None:
        return {'urt_food_name': best_match['urt_food_name'], 'urt': best_match['urt'], 
                'gram_per_portion': best_match['gram_per_portion'], 'similarity_score': best_sim, 'match_type': 'fuzzy_category'}
    
    return None

def apply_category_fuzzy_matching(tkpi_df, urt_df, similarity_threshold=85):
    """Apply category-aware fuzzy matching to merge TKPI with URT."""
    match_results = []
    match_stats = {'exact_name': 0, 'fuzzy_category': 0, 'no_match': 0}

    for _, row in tkpi_df.iterrows():
        result = find_best_urt_match_by_category(row, urt_df, similarity_threshold)
        if result:
            result['match_type_fuzzy'] = result.pop('match_type')
            match_results.append(result)
            match_stats[result['match_type_fuzzy']] += 1
        else:
            match_results.append({'urt_food_name': np.nan, 'urt': np.nan, 'gram_per_portion': np.nan, 
                                  'similarity_score': np.nan, 'match_type_fuzzy': 'no_match'})
            match_stats['no_match'] += 1

    merged = pd.concat([tkpi_df.reset_index(drop=True), pd.DataFrame(match_results)], axis=1)
    return merged, match_stats

# ============================================================================
# CORE HELPER FUNCTIONS
# ============================================================================
def normalize_name(name):
    """Normalize by removing descriptors, punctuation, and SORTING words."""
    if pd.isna(name): return ""
    name = str(name).lower()
    
    # 1. Remove descriptors using word boundaries
    descriptors = [r'\bsegar\b', r'\bmentah\b', r'\brebus\b', r'\bkukus\b', 
                   r'\bkering\b', r'\bgoreng\b', r'\btepung\b', r'\bmasakan\b']
    for desc in descriptors:
        name = re.sub(desc, "", name)
        
    # 2. Remove punctuation (commas, slashes, dashes)
    name = re.sub(r'[^\w\s]', ' ', name)
    
    # 3. Split into words, remove empty spaces, SORT alphabetically, and join
    words = [w for w in name.split() if w]
    words.sort()
    
    return " ".join(words)

def assign_category_with_source(food_code, food_name):
    """Assign food category using TKPI code prefixes and keyword rules."""
    if pd.isna(food_code) or pd.isna(food_name): return (None, "missing_data")
    
    prefix = str(food_code).upper()[:2]
    name = str(food_name).lower()
    
    code_map = {'AR': 'MP', 'AP': 'MP', 'BR': 'MP', 'BP': 'MP', 'CR': 'LN', 'CP': 'LN', 'DR': 'S', 
                'DP': 'S', 'ER': 'B', 'FR': 'LH', 'FP': 'LH', 'GR': 'LH', 'GP': 'LH', 'HR': 'LH', 'JR': 'SS'}
    if prefix in code_map: return (code_map[prefix], 'code_prefix')

    if any(kw in name for kw in ['gula', 'madu', 'sirup', 'selai', 'jam']): return ('G', 'keyword_sugar')
    if any(kw in name for kw in ['minyak', 'lemak', 'margarin', 'mentega']): return ('M', 'keyword_oil')
    return (None, 'unclassified')

def parse_gram(gram_str):
    """Parse URT gram values including ranges (e.g., '125-140' -> 132.5)."""
    if pd.isna(gram_str) or str(gram_str).strip() in ['', '-', '–', '—', 'NA', 'NaN', 'nan']: return np.nan
    gram_str = str(gram_str).strip()
    
    try: return float(gram_str)
    except ValueError: pass

    for delimiter in ['-', '/']:
        if delimiter in gram_str and not gram_str.startswith('-'):
            parts = gram_str.split(delimiter)
            if len(parts) == 2:
                try: return (float(parts[0].strip()) + float(parts[1].strip())) / 2
                except ValueError: return np.nan
    return np.nan

def classify_processing_status(food_name):
    """Classify food as 'Olahan' or 'Non-Olahan' based on keywords."""
    if pd.isna(food_name): return "Unknown"
    keywords = ['asin', 'asap', 'dendeng', 'kaleng', 'sarden', 'sardencis', 'kornet', 'sosis', 
                'ham', 'bakso', 'nugget', 'sirup', 'selai', 'jam', 'madu', 'kerupuk', 'biskuit']
    return "Olahan" if any(kw in str(food_name).lower() for kw in keywords) else "Non-Olahan"

def identify_restricted_processed(food_name, sodium_mg):
    """Flag highly processed/restricted foods based on additives or extreme sodium."""
    if pd.isna(food_name): return False
    keywords = ['asin', 'asap', 'sarden', 'sardencis', 'kornet', 'sosis', 'ham', 'bakso', 'nugget', 'kerupuk']
    if any(kw in str(food_name).lower() for kw in keywords): return True
    return pd.notna(sodium_mg) and sodium_mg > 1000

def assign_smart_portion(food_name, category):
    """Assign standard default portions for common foods missing URT."""
    if pd.isna(food_name): return (None, None)
    name = str(food_name).lower()

    if any(k in name for k in ['nasi', 'beras', 'rice']): return ('3/4 Gelas', 75.0 if 'mentah' in name else 150.0)
    if any(k in name for k in ['telur', 'egg']): return ('1 Butir', 50.0)
    if 'tempe' in name: return ('1 Potong', 100.0)
    if 'tahu' in name or 'tofu' in name: return ('1/2 Piring', 100.0)
    if any(k in name for k in ['ayam', 'daging', 'sapi', 'chicken', 'beef']):
        return ('2 Potong', 70.0) if any(x in name for x in ['rebus', 'goreng', 'kukus']) else ('100 gram', 100.0)
    if any(k in name for k in ['ikan', 'fish']): return ('1 Potong Sedang', 80.0) if 'segar' in name else ('2 Potong', 50.0)
    if category == 'S': return ('1 Gelas', 100.0)
    if any(k in name for k in ['kacang', 'bean']): return ('1/2 Gelas', 75.0)
    return (None, None)

def create_missing_summary(df, columns, output_path):
    """Generate and save missing value summaries."""
    summary = [{'column': col, 'missing_count': df[col].isna().sum(), 
                'total_records': len(df), 'missing_percentage': (df[col].isna().sum() / len(df)) * 100} 
               for col in columns if col in df.columns]
    pd.DataFrame(summary).to_csv(output_path, index=False)

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    print("\n" + "="*80 + "\nDIETARY MENU RECOMMENDATION SYSTEM - DATA PREPROCESSING\n" + "="*80)
    
    dirs = {"raw": Path("raw"), "out": Path("outputs")}
    dirs["out"].mkdir(parents=True, exist_ok=True)
    
    # 1. Load Data
    tkpi = pd.read_csv(dirs["raw"] / "tkpi_raw.csv", dtype=str)
    urt = pd.read_csv(dirs["raw"] / "urt_raw.csv", dtype=str)
    
    # 2. Rename Columns
    tkpi = tkpi.rename(columns={'KODE': 'food_code', 'NAMA BAHAN': 'food_name', 'SUMBER': 'source', 'AIR (g)': 'water_g_100g', 
                                'ENERGI (Kal)': 'energy_kcal_100g', 'PROTEIN (g)': 'protein_g_100g', 'LEMAK (g)': 'fat_g_100g', 
                                'KH (g)': 'carb_g_100g', 'SERAT (g)': 'fiber_g_100g', 'KALSIUM (mg)': 'calcium_mg_100g', 
                                'NATRIUM (mg)': 'sodium_mg_100g', 'KALIUM (mg)': 'potassium_mg_100g', 'BDD (%)': 'bdd_percent'})
    urt = urt.rename(columns={'Nama Pangan': 'urt_food_name', 'Berat dalam Gram': 'urt_gram', 'Ukuran Rumah Tangga (URT)': 'urt'})

    # 3-4. Normalize & Assign Categories
    tkpi['food_name_normalized'] = tkpi['food_name'].apply(normalize_name)
    urt['urt_food_name_normalized'] = urt['urt_food_name'].apply(normalize_name)
    tkpi[['category', 'category_source']] = tkpi.apply(lambda r: pd.Series(assign_category_with_source(r['food_code'], r['food_name_normalized'])), axis=1)
    urt[['category', 'category_source']] = urt.apply(lambda r: pd.Series(assign_category_with_source(None, r['urt_food_name_normalized'])), axis=1)

    # 5-7. Clean Numerics & Parse Grams
    num_cols = ['energy_kcal_100g', 'protein_g_100g', 'fat_g_100g', 'carb_g_100g', 'sodium_mg_100g', 'potassium_mg_100g', 'calcium_mg_100g', 'bdd_percent']
    tkpi[num_cols] = tkpi[num_cols].replace(['', '-', '–', '—', ' ', 'NA', 'NaN', 'nan'], np.nan).apply(pd.to_numeric, errors='coerce')
    if 'urt_gram' in urt.columns: urt['gram_per_portion'] = urt['urt_gram'].apply(parse_gram)

    # 8. Deduplicate URT
    urt = urt.drop_duplicates(subset=['urt_food_name_normalized'], keep='first')

    # 9. Merge using Category-Aware Fuzzy Matching (FIXED)
    print("\n[*] Applying Fuzzy Matching Pipeline...")
    merged, stats = apply_category_fuzzy_matching(tkpi, urt, similarity_threshold=85)
    print(f"    Exact: {stats['exact_name']} | Fuzzy: {stats['fuzzy_category']} | No Match: {stats['no_match']}")

    # 10. Process Matches & Usability
    def map_usability(row):
        if row['category'] == 'S': return pd.Series(['category_standard_vegetable', 100.0, '1 gelas'])
        if pd.notna(row['gram_per_portion']) and pd.notna(row['urt']): return pd.Series(['normalized_urt', row['gram_per_portion'], row['urt']])
        smart_urt, smart_g = assign_smart_portion(row['food_name_normalized'], row['category'])
        if smart_urt: return pd.Series(['smart_portion_default', smart_g, smart_urt])
        return pd.Series(['unmatched', np.nan, np.nan])

    merged[['match_type', 'gram_per_portion', 'URT']] = merged.apply(map_usability, axis=1)
    merged['is_usable_for_recommendation'] = merged['match_type'].isin(['normalized_urt', 'category_standard_vegetable', 'smart_portion_default'])

    # 11-12. Calc Nutrients & Classify Status
    for c in ['energy_kcal', 'protein_g', 'fat_g', 'carb_g', 'sodium_mg', 'potassium_mg', 'calcium_mg']:
        merged[f'{c}_portion'] = (merged['gram_per_portion'] / 100) * merged[f'{c}_100g']
        
    merged['processing_status'] = merged['food_name_normalized'].apply(classify_processing_status)
    merged['restricted_processed_flag'] = merged.apply(lambda r: identify_restricted_processed(r['food_name_normalized'], r['sodium_mg_100g']), axis=1)
    merged['macro_count'] = merged[['energy_kcal_100g', 'protein_g_100g', 'fat_g_100g', 'carb_g_100g']].notna().sum(axis=1)

    # 13-18. Exports
    macro_cols = ['energy_kcal_100g', 'protein_g_100g', 'fat_g_100g', 'carb_g_100g', 'URT', 'gram_per_portion']
    create_missing_summary(merged, macro_cols + ['sodium_mg_100g'], dirs["out"] / "missing_summary_master.csv")
    
    merged.groupby('match_type', dropna=False).agg(total_count=('food_code', 'count'), usable_count=('is_usable_for_recommendation', 'sum')).to_csv(dirs["out"] / "summary_match_type.csv")
    merged[merged['match_type'] == 'unmatched'][['food_code', 'food_name', 'category']].to_csv(dirs["out"] / "unmatched_foods.csv", index=False)

    merged.to_csv(dirs["out"] / "cleaned_tkpi_urt_dataset.csv", index=False)
    
    candidates_mask = (merged['is_usable_for_recommendation']) & (merged['macro_count'] >= 3) & (merged['URT'].notna()) & (merged['gram_per_portion'].notna())
    general_cands = merged[candidates_mask]
    general_cands.to_csv(dirs["out"] / "final_food_candidates.csv", index=False)
    general_cands[~general_cands['restricted_processed_flag']].to_csv(dirs["out"] / "final_food_candidates_clinical.csv", index=False)
    
    print("\n[✓] Preprocessing complete. Files saved to /outputs directory.\n")

if __name__ == "__main__":
    main()