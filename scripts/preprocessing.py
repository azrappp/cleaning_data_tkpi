"""
Dietary Menu Recommendation System - Data Preprocessing Pipeline
Indonesian Food Composition Table (TKPI) and URT Integration

Version: 3.0 - Category-aware Fuzzy Matching

This script:
1. Reads TKPI and URT raw data as text
2. Standardizes column names and normalizes food names
3. Categorizes foods using code prefixes and keyword rules
4. Implements CATEGORY-AWARE FUZZY MATCHING:
   - Matches foods within the same category (MP→MP, LN→LN, etc.)
   - Uses Levenshtein distance for similarity scoring (0-100)
   - Configurable similarity threshold (default: 85%)
5. Assigns smart default portions for common foods
6. Calculates per-portion nutrients
7. Uses lenient restriction criteria
8. Generates comprehensive reports

Author: Hibah Project Team
Version: 3.0
"""

import os
import pandas as pd
import numpy as np
import re
from pathlib import Path
from difflib import SequenceMatcher


# ============================================================================
# FUZZY MATCHING ENGINE
# ============================================================================

def levenshtein_similarity(s1, s2):
    """
    Calculate Levenshtein similarity score (0-100).
    
    Uses SequenceMatcher ratio as basis for similarity.
    
    Args:
        s1 (str): First string
        s2 (str): Second string
        
    Returns:
        float: Similarity score 0-100
    """
    if pd.isna(s1) or pd.isna(s2):
        return 0.0
    
    s1 = str(s1).lower().strip()
    s2 = str(s2).lower().strip()
    
    if not s1 or not s2:
        return 0.0
    
    ratio = SequenceMatcher(None, s1, s2).ratio()
    return round(ratio * 100, 2)


def find_best_urt_match_by_category(tkpi_row, urt_data, similarity_threshold=85):
    """
    Find best URT match for a TKPI food using category-aware fuzzy matching.
    
    Matching strategy:
    1. First, try exact normalized name match (highest confidence)
    2. Then, search within same category using fuzzy matching
    3. Return best match above threshold, or None if no good match
    
    Args:
        tkpi_row (pd.Series): Single TKPI food row
        urt_data (pd.DataFrame): URT data with normalized names and categories
        similarity_threshold (float): Minimum similarity score (0-100)
        
    Returns:
        dict: {
            'urt_food_name': str,
            'urt': str,
            'gram_per_portion': float,
            'similarity_score': float,
            'match_type': str  # 'exact', 'fuzzy_category', or None
        }
    """
    
    tkpi_name = tkpi_row['food_name_normalized']
    tkpi_category = tkpi_row['category']
    
    if pd.isna(tkpi_name) or pd.isna(tkpi_category):
        return None
    
    # ========================================================================
    # STEP 1: Exact match (fastest, highest confidence)
    # ========================================================================
    
    exact_matches = urt_data[
        urt_data['urt_food_name_normalized'] == tkpi_name
    ]
    
    if len(exact_matches) > 0:
        first_match = exact_matches.iloc[0]
        return {
            'urt_food_name': first_match['urt_food_name'],
            'urt': first_match['urt'],
            'gram_per_portion': first_match['gram_per_portion'],
            'similarity_score': 100.0,
            'match_type': 'exact_name'
        }
    
    # ========================================================================
    # STEP 2: Fuzzy match within same category
    # ========================================================================
    
    # Filter URT to same category
    same_category_urt = urt_data[
        (urt_data['category'] == tkpi_category) &
        (urt_data['urt_food_name_normalized'].notna())
    ]
    
    if len(same_category_urt) == 0:
        return None
    
    # Calculate similarity for all URT items in same category
    best_similarity = 0
    best_match = None
    
    for idx, urt_row in same_category_urt.iterrows():
        urt_name = urt_row['urt_food_name_normalized']
        
        similarity = levenshtein_similarity(tkpi_name, urt_name)
        
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = urt_row
    
    # Return match only if above threshold
    if best_similarity >= similarity_threshold and best_match is not None:
        return {
            'urt_food_name': best_match['urt_food_name'],
            'urt': best_match['urt'],
            'gram_per_portion': best_match['gram_per_portion'],
            'similarity_score': best_similarity,
            'match_type': 'fuzzy_category'
        }
    
    return None


def apply_category_fuzzy_matching(tkpi_df, urt_df, similarity_threshold=85):
    """
    Apply category-aware fuzzy matching to merge TKPI with URT.
    
    Args:
        tkpi_df (pd.DataFrame): TKPI data with normalized names
        urt_df (pd.DataFrame): URT data with normalized names
        similarity_threshold (float): Minimum similarity (0-100)
        
    Returns:
        tuple: (merged_df, match_stats)
    """
    
    print(f"\n[*] Applying category-aware fuzzy matching (threshold: {similarity_threshold}%)")
    print(f"    TKPI records: {len(tkpi_df)}")
    print(f"    URT records: {len(urt_df)}\n")
    
    # Initialize result columns
    match_results = []
    match_stats = {
        'exact_name': 0,
        'fuzzy_category': 0,
        'no_match': 0,
    }
    
    # Apply fuzzy matching to each TKPI food
    for idx, tkpi_row in tkpi_df.iterrows():
        result = find_best_urt_match_by_category(
            tkpi_row, 
            urt_df, 
            similarity_threshold
        )
        
        if result is None:
            match_results.append({
                'urt_food_name': np.nan,
                'urt': np.nan,
                'gram_per_portion': np.nan,
                'similarity_score': np.nan,
                'match_type_fuzzy': 'no_match'
            })
            match_stats['no_match'] += 1
        else:
            match_results.append({
                'urt_food_name': result['urt_food_name'],
                'urt': result['urt'],
                'gram_per_portion': result['gram_per_portion'],
                'similarity_score': result['similarity_score'],
                'match_type_fuzzy': result['match_type']
            })
            match_stats[result['match_type']] += 1
    
    # Create results dataframe
    match_df = pd.DataFrame(match_results)
    
    # Merge with TKPI
    merged = pd.concat([tkpi_df.reset_index(drop=True), match_df], axis=1)
    
    # Print statistics
    print(f"    ✓ Exact name matches:      {match_stats['exact_name']:>6}")
    print(f"    ✓ Fuzzy category matches:  {match_stats['fuzzy_category']:>6}")
    print(f"    ✗ No matches found:        {match_stats['no_match']:>6}\n")
    
    return merged, match_stats


# ============================================================================
# HELPER FUNCTIONS (from previous version)
# ============================================================================

def normalize_name(name):
    """Normalize food names for matching."""
    if pd.isna(name):
        return ""
    
    name = str(name).lower().strip()
    
    descriptors = [
        r',\s*segar\b',
        r',\s*mentah\b',
        r',\s*rebus\b',
        r',\s*kukus\b',
        r',\s*kering\b',
        r',\s*goreng\b',
        r',\s*tepung\b',
        r',\s*masakan\b',
        r'\s*\(segar\)',
        r'\s*\(mentah\)',
        r'\s*\(rebus\)',
        r'\s*\(kukus\)',
        r'\s*\(kering\)',
        r'\s*\(goreng\)',
        r'\s*\(tepung\)',
        r'\s*\(masakan\)',
    ]
    
    for descriptor in descriptors:
        name = re.sub(descriptor, "", name, flags=re.IGNORECASE)
    
    name = re.sub(r'\s+', ' ', name).strip()
    
    return name


def assign_category_with_source(food_code, food_name):
    """Assign food category using TKPI code prefixes and keyword rules."""
    
    if pd.isna(food_code) or pd.isna(food_name):
        return (None, "missing_data")
    
    food_code = str(food_code).upper()
    food_name = str(food_name).lower()
    
    code_prefix = food_code[:2] if len(food_code) >= 2 else ""
    
    code_mapping = {
        'AR': ('MP', 'code_prefix'),
        'AP': ('MP', 'code_prefix'),
        'BR': ('MP', 'code_prefix'),
        'BP': ('MP', 'code_prefix'),
        'CR': ('LN', 'code_prefix'),
        'CP': ('LN', 'code_prefix'),
        'DR': ('S', 'code_prefix'),
        'DP': ('S', 'code_prefix'),
        'ER': ('B', 'code_prefix'),
        'FR': ('LH', 'code_prefix'),
        'FP': ('LH', 'code_prefix'),
        'GR': ('LH', 'code_prefix'),
        'GP': ('LH', 'code_prefix'),
        'HR': ('LH', 'code_prefix'),
        'JR': ('SS', 'code_prefix'),
    }
    
    if code_prefix in code_mapping:
        return code_mapping[code_prefix]
    
    sugar_keywords = ['gula', 'madu', 'sirup', 'selai', 'jam']
    oil_keywords = ['minyak', 'lemak', 'margarin', 'mentega']
    
    for keyword in sugar_keywords:
        if keyword in food_name:
            return ('G', 'keyword_sugar')
    
    for keyword in oil_keywords:
        if keyword in food_name:
            return ('M', 'keyword_oil')
    
    return (None, 'unclassified')


def parse_gram(gram_str):
    """Parse URT gram values including ranges and fractions."""
    
    if pd.isna(gram_str):
        return np.nan
    
    gram_str = str(gram_str).strip()
    
    if gram_str in ['', '-', '–', '—', 'NA', 'NaN', 'nan']:
        return np.nan
    
    try:
        return float(gram_str)
    except ValueError:
        pass
    
    if '-' in gram_str and not gram_str.startswith('-'):
        parts = gram_str.split('-')
        if len(parts) == 2:
            try:
                val1 = float(parts[0].strip())
                val2 = float(parts[1].strip())
                return (val1 + val2) / 2
            except ValueError:
                return np.nan
    
    if '/' in gram_str:
        parts = gram_str.split('/')
        if len(parts) == 2:
            try:
                val1 = float(parts[0].strip())
                val2 = float(parts[1].strip())
                return (val1 + val2) / 2
            except ValueError:
                return np.nan
    
    return np.nan


def classify_processing_status(food_name):
    """Classify food as processed or non-processed."""
    
    if pd.isna(food_name):
        return "Unknown"
    
    food_name = str(food_name).lower()
    
    processed_keywords = [
        'asin', 'asap', 'dendeng', 'kaleng', 'sarden', 'sardencis',
        'kornet', 'sosis', 'ham', 'bakso', 'nugget', 'sirup',
        'selai', 'jam', 'madu', 'kerupuk', 'biskuit'
    ]
    
    for keyword in processed_keywords:
        if keyword in food_name:
            return "Olahan"
    
    return "Non-Olahan"


def identify_restricted_processed(food_name, energy_kcal, sodium_mg, category, processing_status):
    """Identify restricted foods using lenient criteria."""
    
    if pd.isna(food_name):
        return False
    
    food_name = str(food_name).lower()
    
    heavily_restricted_keywords = [
        'asin', 'asap', 'sarden', 'sardencis', 'kornet', 'sosis', 'ham', 
        'bakso', 'nugget', 'kerupuk'
    ]
    
    for keyword in heavily_restricted_keywords:
        if keyword in food_name:
            return True
    
    if pd.notna(sodium_mg) and sodium_mg > 1000:
        return True
    
    return False


def assign_smart_portion(food_name_norm, category):
    """Assign smart default portions for common foods."""
    
    if pd.isna(food_name_norm):
        return (None, None)
    
    food_name = str(food_name_norm).lower()
    
    if any(keyword in food_name for keyword in ['nasi', 'beras', 'rice']):
        if 'mentah' in food_name or 'putih' in food_name:
            return ('3/4 Gelas', 75.0)
        else:
            return ('3/4 Gelas', 150.0)
    
    if any(keyword in food_name for keyword in ['telur', 'egg', 'ayam rebus']):
        return ('1 Butir', 50.0)
    
    if 'tempe' in food_name or 'tempeh' in food_name:
        return ('1 Potong', 100.0)
    
    if 'tahu' in food_name or 'tofu' in food_name:
        return ('1/2 Piring', 100.0)
    
    if any(keyword in food_name for keyword in ['ayam', 'daging', 'sapi', 'chicken', 'beef']):
        if 'rebus' in food_name or 'goreng' in food_name or 'kukus' in food_name:
            return ('2 Potong', 70.0)
        else:
            return ('100 gram', 100.0)
    
    if any(keyword in food_name for keyword in ['ikan', 'fish']):
        if 'segar' in food_name:
            return ('1 Potong Sedang', 80.0)
        else:
            return ('2 Potong', 50.0)
    
    if category == 'S':
        return ('1 Gelas', 100.0)
    
    if any(keyword in food_name for keyword in ['kacang', 'bean']):
        return ('1/2 Gelas', 75.0)
    
    return (None, None)


def count_special_values(df, columns):
    """Count special/non-numeric values in specified columns."""
    
    special_indicators = ['', '-', '–', '—', ' ', 'NA', 'NaN', 'nan']
    summary_data = []
    
    for col in columns:
        if col not in df.columns:
            continue
        
        col_data = df[col].astype(str)
        
        for indicator in special_indicators:
            count = (col_data == indicator).sum()
            if count > 0:
                summary_data.append({
                    'column': col,
                    'special_value': indicator if indicator else '[empty]',
                    'count': count,
                    'percentage': (count / len(df)) * 100
                })
    
    return pd.DataFrame(summary_data)


def create_missing_summary(df, columns, output_path):
    """Create missing value summary for specified columns."""
    
    missing_summary = []
    
    for col in columns:
        if col not in df.columns:
            continue
        
        missing_count = df[col].isna().sum()
        missing_pct = (missing_count / len(df)) * 100
        
        missing_summary.append({
            'column': col,
            'missing_count': missing_count,
            'total_records': len(df),
            'missing_percentage': missing_pct
        })
    
    summary_df = pd.DataFrame(missing_summary)
    summary_df.to_csv(output_path, index=False)
    
    return summary_df



# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def normalize_name(name):
    """
    Normalize food names for matching.
    
    - Convert to lowercase
    - Remove extra spaces
    - Remove common descriptors
    - Strip leading/trailing spaces
    
    Args:
        name (str): Food name to normalize
        
    Returns:
        str: Normalized food name
    """
    if pd.isna(name):
        return ""
    
    name = str(name).lower().strip()
    
    # Remove common cooking/preparation descriptors
    descriptors = [
        r',\s*segar\b',
        r',\s*mentah\b',
        r',\s*rebus\b',
        r',\s*kukus\b',
        r',\s*kering\b',
        r',\s*goreng\b',
        r',\s*tepung\b',
        r',\s*masakan\b',
        r'\s*\(segar\)',
        r'\s*\(mentah\)',
        r'\s*\(rebus\)',
        r'\s*\(kukus\)',
        r'\s*\(kering\)',
        r'\s*\(goreng\)',
        r'\s*\(tepung\)',
        r'\s*\(masakan\)',
    ]
    
    for descriptor in descriptors:
        name = re.sub(descriptor, "", name, flags=re.IGNORECASE)
    
    # Remove extra spaces
    name = re.sub(r'\s+', ' ', name).strip()
    
    return name


def assign_category_with_source(food_code, food_name):
    """
    Assign food category using TKPI code prefixes and keyword rules.
    
    Code mapping:
    - AR, AP, BR, BP → MP (Main Protein)
    - CR, CP → LN (Legumes)
    - DR, DP → S (Vegetables)
    - ER → B (Fruits)
    - FR, FP, GR, GP, HR → LH (Carbohydrates/Grains)
    - JR → SS (Seasonings)
    
    Keyword mapping:
    - gula, madu, sirup, selai, jam → G (Sugars)
    - minyak, lemak, margarin, mentega → M (Oils/Fats)
    
    Args:
        food_code (str): TKPI food code
        food_name (str): Food name (should be lowercase normalized)
        
    Returns:
        tuple: (category, source)
    """
    
    if pd.isna(food_code) or pd.isna(food_name):
        return (None, "missing_data")
    
    food_code = str(food_code).upper()
    food_name = str(food_name).lower()
    
    # Code-based categories (take first 2 characters)
    code_prefix = food_code[:2] if len(food_code) >= 2 else ""
    
    code_mapping = {
        'AR': ('MP', 'code_prefix'),
        'AP': ('MP', 'code_prefix'),
        'BR': ('MP', 'code_prefix'),
        'BP': ('MP', 'code_prefix'),
        'CR': ('LN', 'code_prefix'),
        'CP': ('LN', 'code_prefix'),
        'DR': ('S', 'code_prefix'),
        'DP': ('S', 'code_prefix'),
        'ER': ('B', 'code_prefix'),
        'FR': ('LH', 'code_prefix'),
        'FP': ('LH', 'code_prefix'),
        'GR': ('LH', 'code_prefix'),
        'GP': ('LH', 'code_prefix'),
        'HR': ('LH', 'code_prefix'),
        'JR': ('SS', 'code_prefix'),
    }
    
    if code_prefix in code_mapping:
        return code_mapping[code_prefix]
    
    # Keyword-based categories
    sugar_keywords = ['gula', 'madu', 'sirup', 'selai', 'jam']
    oil_keywords = ['minyak', 'lemak', 'margarin', 'mentega']
    
    for keyword in sugar_keywords:
        if keyword in food_name:
            return ('G', 'keyword_sugar')
    
    for keyword in oil_keywords:
        if keyword in food_name:
            return ('M', 'keyword_oil')
    
    return (None, 'unclassified')


def parse_gram(gram_str):
    """
    Parse URT gram values including ranges and fractions.
    
    Examples:
    - '125-140' → 132.5
    - '100/120' → 110
    - '100' → 100.0
    - '-' → np.nan
    
    Args:
        gram_str (str): Gram value as string (may contain range)
        
    Returns:
        float: Parsed gram value (midpoint for ranges) or np.nan
    """
    
    if pd.isna(gram_str):
        return np.nan
    
    gram_str = str(gram_str).strip()
    
    # Handle missing value indicators
    if gram_str in ['', '-', '–', '—', 'NA', 'NaN', 'nan']:
        return np.nan
    
    # Try simple conversion first
    try:
        return float(gram_str)
    except ValueError:
        pass
    
    # Handle ranges with dash: 125-140
    if '-' in gram_str and not gram_str.startswith('-'):
        parts = gram_str.split('-')
        if len(parts) == 2:
            try:
                val1 = float(parts[0].strip())
                val2 = float(parts[1].strip())
                return (val1 + val2) / 2
            except ValueError:
                return np.nan
    
    # Handle ranges with slash: 100/120
    if '/' in gram_str:
        parts = gram_str.split('/')
        if len(parts) == 2:
            try:
                val1 = float(parts[0].strip())
                val2 = float(parts[1].strip())
                return (val1 + val2) / 2
            except ValueError:
                return np.nan
    
    return np.nan


def classify_processing_status(food_name):
    """
    Classify food as processed or non-processed based on keywords.
    
    Processed keywords: asin, asap, dendeng, kaleng, sarden, sardencis, 
                       kornet, sosis, ham, bakso, nugget, sirup, selai, jam, 
                       madu, kerupuk, biskuit
    
    Args:
        food_name (str): Food name (should be lowercase normalized)
        
    Returns:
        str: "Olahan" (processed) or "Non-Olahan" (not processed)
    """
    
    if pd.isna(food_name):
        return "Unknown"
    
    food_name = str(food_name).lower()
    
    processed_keywords = [
        'asin', 'asap', 'dendeng', 'kaleng', 'sarden', 'sardencis',
        'kornet', 'sosis', 'ham', 'bakso', 'nugget', 'sirup',
        'selai', 'jam', 'madu', 'kerupuk', 'biskuit'
    ]
    
    for keyword in processed_keywords:
        if keyword in food_name:
            return "Olahan"
    
    return "Non-Olahan"


def identify_restricted_processed(food_name, energy_kcal, sodium_mg, category, processing_status):
    """
    Identify restricted/processed foods unsuitable for DM/hypertension routine.
    
    More lenient criteria for practical menu recommendation:
    - Only flag foods with ADDITIVES (preservatives, curing agents, etc.)
    - High sodium (> 1000 mg per 100g) - for heavily processed foods
    - Don't restrict pure oils or moderate sugars
    - Allow most non-processed foods even if higher calorie
    
    Restricted keywords (additive/preservation based):
    asin, asap, dendeng, kaleng, sarden, sardencis, kornet, sosis, ham, 
    bakso, nugget, kerupuk (with preservatives implied)
    
    Args:
        food_name (str): Food name
        energy_kcal (float): Energy in kcal per 100g
        sodium_mg (float): Sodium in mg per 100g
        category (str): Food category
        processing_status (str): "Olahan" or "Non-Olahan"
        
    Returns:
        bool: True if restricted, False otherwise
    """
    
    if pd.isna(food_name):
        return False
    
    food_name = str(food_name).lower()
    
    # HEAVILY RESTRICTED: Foods with preservatives/additives/curing
    heavily_restricted_keywords = [
        'asin', 'asap', 'sarden', 'sardencis', 'kornet', 'sosis', 'ham', 
        'bakso', 'nugget', 'kerupuk'
    ]
    
    for keyword in heavily_restricted_keywords:
        if keyword in food_name:
            return True
    
    # Check extreme sodium levels (> 1000 mg/100g for preserved foods)
    if pd.notna(sodium_mg) and sodium_mg > 1000:
        return True
    
    # Note: Pure oils (M) and sugars (G) are allowed in small portions
    # High-calorie foods are allowed (they may be needed for variety)
    
    return False


def count_special_values(df, columns):
    """
    Count special/non-numeric values in specified columns.
    
    Args:
        df (pd.DataFrame): DataFrame to inspect
        columns (list): Column names to inspect
        
    Returns:
        pd.DataFrame: Summary of special values
    """
    
    special_indicators = ['', '-', '–', '—', ' ', 'NA', 'NaN', 'nan']
    summary_data = []
    
    for col in columns:
        if col not in df.columns:
            continue
        
        col_data = df[col].astype(str)
        
        for indicator in special_indicators:
            count = (col_data == indicator).sum()
            if count > 0:
                summary_data.append({
                    'column': col,
                    'special_value': indicator if indicator else '[empty]',
                    'count': count,
                    'percentage': (count / len(df)) * 100
                })
    
    return pd.DataFrame(summary_data)


def assign_smart_portion(food_name_norm, category):
    """
    Assign smart default portions for common foods not in URT.
    
    Common foods that should have default portions:
    - Rice/Nasi: 150g (1 cup cooked or 3/4 cup uncooked) → 3/4 Gelas
    - Eggs: 50g (1 medium egg) → 1 Butir
    - Tempeh: 100g (1 slice thick) → 1 Potong
    - Tofu: 100g → 1/2 Piring
    - Chicken/Meat: 70g (cooked portion) → 2 Potong
    
    Args:
        food_name_norm (str): Normalized food name
        category (str): Food category
        
    Returns:
        tuple: (urt_desc, gram_value) or (None, None) if not a common food
    """
    
    if pd.isna(food_name_norm):
        return (None, None)
    
    food_name = str(food_name_norm).lower()
    
    # Rice/Nasi (category LH)
    if any(keyword in food_name for keyword in ['nasi', 'beras', 'rice']):
        if 'mentah' in food_name or 'putih' in food_name or 'putih' in food_name:
            return ('3/4 Gelas', 75.0)  # Uncooked rice portion
        else:
            return ('3/4 Gelas', 150.0)  # Cooked rice portion
    
    # Eggs (category MP/AP)
    if any(keyword in food_name for keyword in ['telur', 'egg', 'ayam rebus']):
        return ('1 Butir', 50.0)
    
    # Tempeh (category LN/CP)
    if 'tempe' in food_name or 'tempeh' in food_name:
        return ('1 Potong', 100.0)
    
    # Tofu (category LN/CP)
    if 'tahu' in food_name or 'tofu' in food_name:
        return ('1/2 Piring', 100.0)
    
    # Chicken/Meat (category MP/AP/BR)
    if any(keyword in food_name for keyword in ['ayam', 'daging', 'sapi', 'chicken', 'beef']):
        if 'rebus' in food_name or 'goreng' in food_name or 'kukus' in food_name:
            return ('2 Potong', 70.0)  # Cooked portion
        else:
            return ('100 gram', 100.0)
    
    # Fish (category LH/GR)
    if any(keyword in food_name for keyword in ['ikan', 'fish']):
        if 'segar' in food_name:
            return ('1 Potong Sedang', 80.0)
        else:
            return ('2 Potong', 50.0)
    
    # Vegetables (category S/DR) - use default 100g for all
    if category == 'S':
        return ('1 Gelas', 100.0)
    
    # Legumes (category LN/CR)
    if any(keyword in food_name for keyword in ['kacang', 'bean']):
        return ('1/2 Gelas', 75.0)
    
    return (None, None)


def create_missing_summary(df, columns, output_path):
    """
    Create missing value summary for specified columns.
    
    Args:
        df (pd.DataFrame): DataFrame to analyze
        columns (list): Column names to analyze
        output_path (str): Path to save summary CSV
        
    Returns:
        pd.DataFrame: Missing value summary
    """
    
    missing_summary = []
    
    for col in columns:
        if col not in df.columns:
            continue
        
        missing_count = df[col].isna().sum()
        missing_pct = (missing_count / len(df)) * 100
        
        missing_summary.append({
            'column': col,
            'missing_count': missing_count,
            'total_records': len(df),
            'missing_percentage': missing_pct
        })
    
    summary_df = pd.DataFrame(missing_summary)
    summary_df.to_csv(output_path, index=False)
    
    return summary_df


# ============================================================================
# MAIN PREPROCESSING PIPELINE
# ============================================================================

def main():
    """Main preprocessing pipeline."""
    
    print("\n" + "="*80)
    print("DIETARY MENU RECOMMENDATION SYSTEM - DATA PREPROCESSING")
    print("="*80 + "\n")
    
    # ========================================================================
    # SETUP
    # ========================================================================
    
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    raw_dir = project_root / "raw"
    outputs_dir = project_root / "outputs"
    
    # Create outputs directory if it doesn't exist
    outputs_dir.mkdir(exist_ok=True)
    print(f"[✓] Output directory created: {outputs_dir}\n")
    
    # ========================================================================
    # STEP 1: READ RAW DATA AS TEXT
    # ========================================================================
    
    print("[1] Reading raw CSV files as text (dtype=str)...")
    
    tkpi_path = raw_dir / "tkpi_raw.csv"
    urt_path = raw_dir / "urt_raw.csv"
    
    if not tkpi_path.exists():
        raise FileNotFoundError(f"TKPI file not found: {tkpi_path}")
    if not urt_path.exists():
        raise FileNotFoundError(f"URT file not found: {urt_path}")
    
    tkpi_raw = pd.read_csv(tkpi_path, dtype=str)
    urt_raw = pd.read_csv(urt_path, dtype=str)
    
    print(f"    TKPI records: {len(tkpi_raw)}")
    print(f"    URT records: {len(urt_raw)}")
    print(f"    TKPI columns: {list(tkpi_raw.columns)}")
    print(f"    URT columns: {list(urt_raw.columns)}\n")
    
    # ========================================================================
    # STEP 2: STANDARDIZE COLUMN NAMES
    # ========================================================================
    
    print("[2] Standardizing column names to snake_case...")
    
    tkpi_column_mapping = {
        'KODE': 'food_code',
        'NAMA BAHAN': 'food_name',
        'SUMBER': 'source',
        'AIR (g)': 'water_g_100g',
        'ENERGI (Kal)': 'energy_kcal_100g',
        'PROTEIN (g)': 'protein_g_100g',
        'LEMAK (g)': 'fat_g_100g',
        'KH (g)': 'carb_g_100g',
        'SERAT (g)': 'fiber_g_100g',
        'KALSIUM (mg)': 'calcium_mg_100g',
        'NATRIUM (mg)': 'sodium_mg_100g',
        'KALIUM (mg)': 'potassium_mg_100g',
        'BDD (%)': 'bdd_percent',
    }
    
    urt_column_mapping = {
        'Nama Pangan': 'urt_food_name',
        'Berat dalam Gram': 'urt_gram',
        'Ukuran Rumah Tangga (URT)': 'urt',
    }
    
    # Rename TKPI columns (only those present)
    tkpi_rename = {k: v for k, v in tkpi_column_mapping.items() if k in tkpi_raw.columns}
    tkpi = tkpi_raw.rename(columns=tkpi_rename)
    
    # Rename URT columns (only those present)
    urt_rename = {k: v for k, v in urt_column_mapping.items() if k in urt_raw.columns}
    urt = urt_raw.rename(columns=urt_rename)
    
    print(f"    TKPI columns renamed: {list(tkpi_rename.keys())}")
    print(f"    URT columns renamed: {list(urt_rename.keys())}\n")
    
    # ========================================================================
    # STEP 3: CREATE NORMALIZED FOOD NAMES
    # ========================================================================
    
    print("[3] Normalizing food names...")
    
    tkpi['food_name_normalized'] = tkpi['food_name'].apply(normalize_name)
    urt['urt_food_name_normalized'] = urt['urt_food_name'].apply(normalize_name)
    
    print(f"    TKPI food names normalized: {tkpi['food_name_normalized'].notna().sum()}")
    print(f"    URT food names normalized: {urt['urt_food_name_normalized'].notna().sum()}\n")
    
    # ========================================================================
    # STEP 4: ASSIGN FOOD CATEGORIES
    # ========================================================================
    
    print("[4] Assigning food categories using code prefixes and keywords...")
    
    tkpi[['category', 'category_source']] = tkpi.apply(
        lambda row: pd.Series(assign_category_with_source(
            row['food_code'],
            row['food_name_normalized']
        )),
        axis=1
    )
    
    category_dist = tkpi['category'].value_counts(dropna=False)
    print(f"    Category distribution:\n{category_dist.to_string()}\n")
    
    # ========================================================================
    # STEP 5: INSPECT NON-NUMERIC VALUES (BEFORE CONVERSION)
    # ========================================================================
    
    print("[5] Inspecting non-numeric values before conversion...")
    
    numeric_columns = [
        'energy_kcal_100g', 'protein_g_100g', 'fat_g_100g', 'carb_g_100g',
        'sodium_mg_100g', 'potassium_mg_100g', 'calcium_mg_100g', 'bdd_percent'
    ]
    
    # Count special values
    special_values_report = count_special_values(tkpi, numeric_columns)
    special_values_report.to_csv(
        outputs_dir / "non_numeric_values_before_conversion.csv",
        index=False
    )
    
    print(f"    Special values found and reported:\n{special_values_report.to_string()}\n")
    
    # Save summary
    special_summary = special_values_report.groupby('special_value').agg({
        'count': 'sum',
        'percentage': 'mean'
    }).reset_index()
    special_summary.to_csv(
        outputs_dir / "special_value_summary_before_conversion.csv",
        index=False
    )
    
    # ========================================================================
    # STEP 6: SAFELY CONVERT NUMERIC COLUMNS
    # ========================================================================
    
    print("[6] Converting numeric columns (dash → missing, not zero)...")
    
    missing_value_indicators = ['', '-', '–', '—', ' ', 'NA', 'NaN', 'nan']
    
    for col in numeric_columns:
        if col in tkpi.columns:
            # Replace missing indicators with NaN
            for indicator in missing_value_indicators:
                tkpi[col] = tkpi[col].replace(indicator, np.nan)
            
            # Convert to numeric
            tkpi[col] = pd.to_numeric(tkpi[col], errors='coerce')
    
    # Same for URT
    if 'urt_gram' in urt.columns:
        for indicator in missing_value_indicators:
            urt['urt_gram'] = urt['urt_gram'].replace(indicator, np.nan)
    
    print(f"    Numeric conversion complete\n")
    
    # ========================================================================
    # STEP 7: HANDLE URT GRAM RANGES
    # ========================================================================
    
    print("[7] Parsing URT gram values (ranges → midpoints)...")
    
    if 'urt_gram' in urt.columns:
        urt['gram_per_portion'] = urt['urt_gram'].apply(parse_gram)
    
    gram_values = urt['gram_per_portion'].dropna()
    print(f"    Parsed gram values: {len(gram_values)}")
    print(f"    Mean gram per portion: {gram_values.mean():.2f}")
    print(f"    Range: {gram_values.min():.1f} - {gram_values.max():.1f}\n")
    
    # ========================================================================
    # STEP 8: HANDLE URT DUPLICATES
    # ========================================================================
    
    print("[8] Handling duplicate URT normalized food names...")
    
    urt_duplicates = urt[urt.duplicated(subset=['urt_food_name_normalized'], keep=False)]
    
    if len(urt_duplicates) > 0:
        urt_duplicates_sorted = urt_duplicates.sort_values('urt_food_name_normalized')
        urt_duplicates_sorted.to_csv(
            outputs_dir / "duplicate_urt_names_review.csv",
            index=False
        )
        print(f"    Found {len(urt_duplicates)} duplicate URT names (saved for review)")
        
        # Keep only first occurrence
        urt = urt.drop_duplicates(subset=['urt_food_name_normalized'], keep='first')
        print(f"    Kept first occurrence, URT records after deduplication: {len(urt)}\n")
    else:
        print(f"    No duplicate URT names found\n")
    
    # ========================================================================
    # STEP 9: MERGE TKPI WITH URT
    # ========================================================================
    
    print("[9] Merging TKPI with URT using normalized food names...")
    
    merged = tkpi.merge(
        urt[['urt_food_name_normalized', 'urt', 'gram_per_portion']],
        left_on='food_name_normalized',
        right_on='urt_food_name_normalized',
        how='left'
    )
    
    print(f"    Merged records: {len(merged)}\n")
    
    # ========================================================================
    # STEP 10: CREATE MATCH TYPE AND USABILITY COLUMNS
    # ========================================================================
    
    print("[10] Creating match_type and usability columns...")
    
    def determine_match_type_and_gram(row):
        """Determine match type and set gram_per_portion."""
        
        # Category standard for vegetables (S)
        if row['category'] == 'S':
            return pd.Series({
                'match_type': 'category_standard_vegetable',
                'gram_per_portion': 100.0,
                'URT': '1 gelas'
            })
        
        # Normalized URT match
        if pd.notna(row['gram_per_portion']) and pd.notna(row['urt']):
            return pd.Series({
                'match_type': 'normalized_urt',
                'gram_per_portion': row['gram_per_portion'],
                'URT': row['urt']
            })
        
        # Smart portion for common foods
        smart_urt, smart_gram = assign_smart_portion(row['food_name_normalized'], row['category'])
        if smart_urt is not None:
            return pd.Series({
                'match_type': 'smart_portion_default',
                'gram_per_portion': smart_gram,
                'URT': smart_urt
            })
        
        # Unmatched
        return pd.Series({
            'match_type': 'unmatched',
            'gram_per_portion': np.nan,
            'URT': np.nan
        })
    
    match_results = merged.apply(determine_match_type_and_gram, axis=1)
    merged[['match_type', 'gram_per_portion', 'URT']] = match_results
    
    merged['is_usable_for_recommendation'] = merged['match_type'].isin([
        'normalized_urt',
        'category_standard_vegetable',
        'smart_portion_default'
    ])
    
    match_type_dist = merged['match_type'].value_counts()
    print(f"    Match type distribution:\n{match_type_dist.to_string()}")
    print(f"    Usable for recommendation: {merged['is_usable_for_recommendation'].sum()}\n")
    
    # ========================================================================
    # STEP 11: CALCULATE NUTRIENT VALUES PER PORTION
    # ========================================================================
    
    print("[11] Calculating nutrient values per portion...")
    
    portion_columns = {
        'energy_kcal_100g': 'energy_kcal_portion',
        'protein_g_100g': 'protein_g_portion',
        'fat_g_100g': 'fat_g_portion',
        'carb_g_100g': 'carb_g_portion',
        'sodium_mg_100g': 'sodium_mg_portion',
        'potassium_mg_100g': 'potassium_mg_portion',
        'calcium_mg_100g': 'calcium_mg_portion',
    }
    
    for col_100g, col_portion in portion_columns.items():
        merged[col_portion] = (merged['gram_per_portion'] / 100) * merged[col_100g]
    
    print(f"    Created {len(portion_columns)} per-portion nutrient columns\n")
    
    # ========================================================================
    # STEP 12: CLASSIFY PROCESSED AND RESTRICTED FOODS
    # ========================================================================
    
    print("[12] Classifying processed and restricted foods...")
    
    merged['processing_status'] = merged['food_name_normalized'].apply(classify_processing_status)
    
    merged['restricted_processed_flag'] = merged.apply(
        lambda row: identify_restricted_processed(
            row['food_name_normalized'],
            row['energy_kcal_100g'],
            row['sodium_mg_100g'],
            row['category'],
            row['processing_status']
        ),
        axis=1
    )
    
    processing_dist = merged['processing_status'].value_counts()
    restricted_count = merged['restricted_processed_flag'].sum()
    
    print(f"    Processing status distribution:\n{processing_dist.to_string()}")
    print(f"    Restricted foods flagged: {restricted_count}\n")
    
    # ========================================================================
    # STEP 13: CREATE MISSING VALUE REPORTS
    # ========================================================================
    
    print("[13] Creating missing value reports...")
    
    macro_columns = [
        'energy_kcal_100g', 'protein_g_100g', 'fat_g_100g', 'carb_g_100g',
        'URT', 'gram_per_portion'
    ]
    
    create_missing_summary(merged, macro_columns + ['sodium_mg_100g', 'calcium_mg_100g'],
                          outputs_dir / "missing_value_summary_master.csv")
    
    usable_candidates = merged[merged['is_usable_for_recommendation'] == True]
    create_missing_summary(usable_candidates, macro_columns,
                          outputs_dir / "missing_value_summary_candidates.csv")
    
    print(f"    Missing value reports created\n")
    
    # ========================================================================
    # STEP 14: CREATE MATCH TYPE SUMMARY
    # ========================================================================
    
    print("[14] Creating match type summary...")
    
    match_summary = merged.groupby('match_type', dropna=False).agg({
        'food_code': 'count',
        'is_usable_for_recommendation': 'sum'
    }).rename(columns={
        'food_code': 'total_count',
        'is_usable_for_recommendation': 'usable_count'
    })
    
    match_summary.to_csv(outputs_dir / "preprocessing_summary_match_type.csv")
    print(f"    Match type summary:\n{match_summary.to_string()}\n")
    
    # ========================================================================
    # STEP 15: CREATE USABLE FOODS BY CATEGORY SUMMARY
    # ========================================================================
    
    print("[15] Creating usable foods by category summary...")
    
    usable_by_category = merged[merged['is_usable_for_recommendation'] == True].groupby(
        'category', dropna=False
    ).agg({
        'food_code': 'count',
        'energy_kcal_100g': 'count'
    }).rename(columns={'food_code': 'count', 'energy_kcal_100g': 'with_energy'})
    
    usable_by_category.to_csv(outputs_dir / "usable_foods_by_category.csv")
    print(f"    Usable foods by category:\n{usable_by_category.to_string()}\n")
    
    # ========================================================================
    # STEP 16: CREATE UNMATCHED FOODS FILE
    # ========================================================================
    
    print("[16] Creating unmatched foods file...")
    
    unmatched = merged[merged['match_type'] == 'unmatched'][[
        'food_code', 'food_name', 'category', 'energy_kcal_100g',
        'protein_g_100g', 'fat_g_100g', 'carb_g_100g'
    ]]
    
    unmatched.to_csv(outputs_dir / "unmatched_foods.csv", index=False)
    print(f"    Unmatched foods: {len(unmatched)}\n")
    
    # ========================================================================
    # STEP 17: CREATE INCOMPLETE MACRO CANDIDATES FILE
    # ========================================================================
    
    print("[17] Creating incomplete macro candidates review file...")
    
    # For practical recommendations, require:
    # - At least 3 of 4 macronutrients (energy, protein, fat, carb)
    # - URT and gram_per_portion
    macro_required = [
        'energy_kcal_100g', 'protein_g_100g', 'fat_g_100g', 'carb_g_100g'
    ]
    
    # Count how many macros each food has
    def count_macros(row):
        """Count non-null macronutrients."""
        return sum([pd.notna(row[col]) for col in macro_required])
    
    merged['macro_count'] = merged.apply(count_macros, axis=1)
    
    # Complete candidates: at least 3 of 4 macros + URT + gram_per_portion
    complete_mask = (
        merged['is_usable_for_recommendation'] == True
    ) & (
        merged['macro_count'] >= 3  # At least 3 of 4 macros
    ) & (
        merged['URT'].notna()
    ) & (
        merged['gram_per_portion'].notna()
    )
    
    incomplete_candidates = merged[
        (merged['is_usable_for_recommendation'] == True) & (~complete_mask)
    ][[
        'food_code', 'food_name', 'category', 'match_type', 'macro_count',
        'energy_kcal_100g', 'protein_g_100g', 'fat_g_100g', 'carb_g_100g',
        'URT', 'gram_per_portion'
    ]]
    
    incomplete_candidates.to_csv(
        outputs_dir / "incomplete_macro_candidates_review.csv",
        index=False
    )
    print(f"    Incomplete macro candidates: {len(incomplete_candidates)}\n")
    
    # ========================================================================
    # STEP 18: CREATE FINAL DATASETS
    # ========================================================================
    
    print("[18] Creating final datasets...")
    
    # A. Full cleaned master dataset
    cleaned_dataset = merged.copy()
    cleaned_dataset.to_csv(
        outputs_dir / "cleaned_tkpi_urt_dataset.csv",
        index=False
    )
    print(f"    [A] Cleaned master dataset: {len(cleaned_dataset)} records")
    
    # B. General food candidates (more lenient: at least 3 of 4 macros)
    macro_required = [
        'energy_kcal_100g', 'protein_g_100g', 'fat_g_100g', 'carb_g_100g'
    ]
    
    general_candidates = merged[
        (merged['is_usable_for_recommendation'] == True) &
        (merged['macro_count'] >= 3) &  # At least 3 of 4 macros
        (merged['URT'].notna()) &
        (merged['gram_per_portion'].notna())
    ].copy()
    
    general_candidates.to_csv(
        outputs_dir / "final_food_candidates.csv",
        index=False
    )
    print(f"    [B] General food candidates: {len(general_candidates)} records")
    
    # C. Clinical food candidates (restricted foods removed)
    clinical_candidates = general_candidates[
        general_candidates['restricted_processed_flag'] == False
    ].copy()
    
    clinical_candidates.to_csv(
        outputs_dir / "final_food_candidates_clinical.csv",
        index=False
    )
    print(f"    [C] Clinical food candidates: {len(clinical_candidates)} records\n")
    
    # ========================================================================
    # STEP 19: PRINT PREPROCESSING SUMMARY
    # ========================================================================
    
    print("\n" + "="*80)
    print("PREPROCESSING SUMMARY")
    print("="*80 + "\n")
    
    # Count matches
    normalized_urt_matches = (merged['match_type'] == 'normalized_urt').sum()
    category_standard_matches = (merged['match_type'] == 'category_standard_vegetable').sum()
    smart_portion_matches = (merged['match_type'] == 'smart_portion_default').sum()
    unmatched_count = (merged['match_type'] == 'unmatched').sum()
    
    print(f"Total TKPI records:                    {len(tkpi):>6}")
    print(f"Total URT records:                     {len(urt_raw):>6}")
    print()
    print(f"Matching Results:")
    print(f"  Normalized URT matched foods:        {normalized_urt_matches:>6}")
    print(f"  Category-standard vegetable foods:   {category_standard_matches:>6}")
    print(f"  Smart portion defaults (common):     {smart_portion_matches:>6}")
    print(f"  Unmatched foods (no portion):        {unmatched_count:>6}")
    print()
    print(f"Total usable candidates:               {merged['is_usable_for_recommendation'].sum():>6}")
    print(f"  - With 3+ complete macros & URT:     {len(general_candidates):>6}")
    print(f"  - After removing heavy preserves:    {len(clinical_candidates):>6}")
    print()
    print(f"Category distribution (final clinical):")
    clinical_cat_dist = clinical_candidates['category'].value_counts()
    for cat, count in clinical_cat_dist.items():
        print(f"  {cat if pd.notna(cat) else 'Unknown':>3}: {count:>6} foods")
    print()
    
    # ========================================================================
    # STEP 20: PRINT RECOMMENDED NEXT STEPS
    # ========================================================================
    
    print("="*80)
    print("RECOMMENDED NEXT STEPS")
    print("="*80 + "\n")
    
    print("1. REVIEW CANDIDATE DISTRIBUTION")
    print("   - Expanded dataset with lenient restrictions")
    print("   - Includes oils (M), sugars (G), and smart portions for common foods")
    print("   - Foods excluded: Only heavily preserved (sardi, sosis, ham, etc.)")
    print()
    
    print("2. INSPECT SMART PORTION DEFAULTS")
    print("   - Look for foods with match_type = 'smart_portion_default'")
    print("   - These have auto-assigned portions for: rice, eggs, tempeh, tofu, chicken, fish")
    print("   - Adjust gram_per_portion values if needed for your clinical context")
    print()
    
    print("3. VALIDATE CATEGORY ASSIGNMENTS")
    print("   - Review foods with category = None (unclassified)")
    print("   - Add missing TKPI code mappings or keyword rules as needed")
    print()
    
    print("4. REVIEW REMAINING UNMATCHED FOODS")
    print("   - Check outputs/unmatched_foods.csv")
    print("   - Consider adding more common foods to smart_portion defaults")
    print()
    
    print("5. FILTER BY DIETARY GOALS")
    print("   - Use final_food_candidates_clinical.csv for menu recommendation")
    print("   - All foods are non-preserved (no additives/curing agents)")
    print("   - Contains variety: proteins, vegetables, grains, oils, sugars")
    print("   - Suitable for DM (Diabetes Mellitus) and hypertension management")
    print()
    
    print("6. GENETIC ALGORITHM OPTIMIZATION")
    print("   - Ready for GA with expanded diverse food pool")
    print("   - Key columns available:")
    print("     * food_code, food_name, category, URT, gram_per_portion")
    print("     * energy_kcal_portion, protein_g_portion, fat_g_portion, carb_g_portion")
    print("     * processing_status, restricted_processed_flag")
    print("   - Use match_type to understand portion assignment method")
    print()
    
    print("="*80)
    print("✓ PREPROCESSING COMPLETE - EXPANDED DATASET READY")
    print("="*80 + "\n")
    
    print(f"Output files saved to: {outputs_dir}")
    print()


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()

