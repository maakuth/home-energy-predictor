# HEPO Battery Power Integration Analysis - Document Index

This directory now contains comprehensive documentation analyzing how load power is calculated in HEPO and what happens when a home battery is added.

## Documents Generated

### 1. **docs/archive/LOAD-CALC-SEARCH-RESULTS.txt** (START HERE)
   - **Purpose:** Executive summary of findings
   - **Length:** ~200 lines
   - **Best For:** Quick understanding of the problem
   - **Key Sections:**
     - Load power calculation breakdown
     - Where grid_power, solar_power, and battery_power interact
     - Battery impact examples with real numbers
     - Critical gaps and missing sensors
     - Summary of findings

   **Read this first if you want a quick overview.**

---

### 2. **SENSOR_POWER_ANALYSIS.md** (DETAILED REFERENCE)
   - **Purpose:** Comprehensive technical deep-dive
   - **Length:** ~400 lines
   - **Best For:** Understanding the architecture in detail
   - **Key Sections:**
     1. Sensor architecture table (all 10+ sensors)
     2. Step-by-step load calculation formula (with physics)
     3. Current assumptions about power flow
     4. Feature engineering in train_model.py
     5. Baseload lag calculation in predict_future.py
     6. The missing piece: Battery power impact
     7. Critical issue: History and model training
     8. README guidance on what should happen
     9. Where battery is currently handled (optimization only)
     10. Summary table of power components
     11. Implications and recommendations
     12. Code locations reference

   **Read this for complete understanding of all components.**

---

### 3. **LOAD_CALCULATION_FLOWCHART.md** (VISUAL REFERENCE)
   - **Purpose:** Visual representation and formula reference
   - **Length:** ~300 lines
   - **Best For:** Understanding data flow and formulas
   - **Key Sections:**
     1. Data flow diagram (Home Assistant → Training → Prediction)
     2. Formula reference card (6 key formulas)
     3. Known issues with current approach (5 issues)
     4. What changes when battery is installed
     5. Testing and validation strategy
     6. Summary power balance table with examples

   **Read this to see diagrams and formula cards.**

---

## Quick Facts

| Question | Answer | Details |
|----------|--------|---------|
| **Is baseload correctly calculated now?** | ✓ YES | Works fine for battery-free systems |
| **Will it break with a battery?** | ✓ YES | Battery power not measured or subtracted |
| **Is battery simulated?** | ✓ YES | In optimize_plan.py (planning only) |
| **Is battery measured historically?** | ✗ NO | No battery_power sensor extracted |
| **What's the critical gap?** | Battery power not in raw_data.csv | Must extract before battery installed |
| **What's the fix?** | 1 line in process_data.py | Add `- battery_power` to formula |
| **Will current tests catch this?** | ✗ NO | Tests assume battery-free system |

---

## The Core Problem

```
WITHOUT BATTERY (Current):
  Home Load = Grid Import + Solar Production ✓ CORRECT

WITH BATTERY (If Not Fixed):
  Home Load = Grid Import + Solar Production ✗ WRONG
  Missing: - Battery Discharge + Battery Charge
```

**Physics:** When a battery charges from solar, it stores energy that shouldn't be counted as home consumption. When it discharges, it's providing power to the home and should be added.

---

## The Formula That Needs Updating

**Current (process_data.py, line 50):**
```python
df['total_home_power'] = df['total_power'] + df['solar_actual']
```

**Needed (with battery):**
```python
df['total_home_power'] = df['total_power'] + df['solar_actual'] - df.get('battery_power', 0)
```

**Where `battery_power` = discharge - charge (positive = discharging)**

---

## What Each Document Covers

### archive/LOAD-CALC-SEARCH-RESULTS.txt
- **Section 1:** Load power calculation (lines 49-61 of process_data.py)
- **Section 2:** Where sensors interact
- **Section 3:** Battery impact examples
- **Section 4:** Model training impact
- **Section 5:** Real-time prediction impact  
- **Section 6:** Battery optimization (simulated only)
- **Section 7:** Performance analysis (uses wrong formula)
- **Appendix:** Assumptions and gaps

### SENSOR_POWER_ANALYSIS.md
- **Section 1:** All 10+ sensors extracted
- **Section 2:** Step-by-step load calculation
- **Section 3:** Power flow assumptions
- **Section 4:** Where load is used (features)
- **Section 5:** Battery impact - THE MISSING PIECE
- **Section 6:** History and model training issues
- **Section 7:** README guidance
- **Section 8:** Battery handling in optimization
- **Section 9:** Missing sensors
- **Section 10:** Summary component table
- **Section 11:** Recommendations
- **Section 12:** Code locations

### LOAD_CALCULATION_FLOWCHART.md
- **Data Flow Diagram:** Home Assistant → Raw → Processed → Model → Predict → Analyze
- **Formula Cards:** 6 key equations with explanations
- **Known Issues:** 5 specific problems identified
- **What Changes:** Before/during/after battery scenarios
- **Testing Strategy:** Unit, integration, and performance monitoring
- **Power Balance Table:** Example scenarios with correct/incorrect calculations

---

## Reading Path by Role

### For Quick Understanding (5 min)
1. Read archive/LOAD-CALC-SEARCH-RESULTS.txt (KEY FINDINGS section)
2. Look at Example Scenarios (Scenario 1 & 2)

### For Code Review (30 min)
1. archive/LOAD-CALC-SEARCH-RESULTS.txt (all sections)
2. SENSOR_POWER_ANALYSIS.md (Sections 2, 4, 5, 12)
3. LOAD_CALCULATION_FLOWCHART.md (Formula Cards)

### For Implementation (1-2 hours)
1. All three documents in order
2. Focus on SENSOR_POWER_ANALYSIS.md sections 6-11
3. Check LOAD_CALCULATION_FLOWCHART.md Testing Strategy

### For Validation (full day)
1. Read all documents completely
2. Cross-reference with actual code files
3. Run tests from LOAD_CALCULATION_FLOWCHART.md
4. Verify performance_analysis metrics

---

## Key Code Files Referenced

| File | Lines | Component |
|------|-------|-----------|
| extract_data.py | 12-28 | Sensor definitions |
| extract_data.py | 35-136 | Data extraction logic |
| process_data.py | 49-52 | total_home_power calculation |
| process_data.py | 54-61 | baseload_power calculation |
| process_data.py | 92-150 | Feature engineering |
| train_model.py | 16-27 | Model target and features |
| predict_future.py | 194-224 | Baseload lag calculation |
| optimize_plan.py | 107-292 | Battery dispatch simulation |
| analyze_performance.py | 40-42 | Actual usage calculation |
| analyze_performance.py | 141-216 | Battery performance eval |

---

## The Integration Timeline

### Current State (No Battery)
- ✅ Extraction: Grid and solar measured correctly
- ✅ Processing: Baseload calculated as grid + solar - known loads
- ✅ Training: Model learns baseload patterns accurately
- ✅ Prediction: Baseload and lag features calculated correctly
- ✅ Optimization: Battery simulated (works on predictions)
- ✅ Analysis: Performance metrics valid (no battery to confound)

### When Battery Installed (Without Code Changes)
- ✅ Extraction: Grid, solar, and AAHP measured (but not battery!)
- ❌ Processing: Baseload calculated WRONG (missing battery term)
- ❌ Training: Model learns INFLATED baseload patterns
- ❌ Prediction: Lag features WRONG, predictions OVERESTIMATE
- ⚠️ Optimization: Battery simulated on WRONG predicted load
- ❌ Analysis: Metrics BIASED due to wrong actual_usage

### When Battery Installed (With Code Changes)
- ✅ Extraction: Grid, solar, battery, AAHP all measured
- ✅ Processing: Baseload calculated CORRECTLY (battery subtracted)
- ✅ Training: Model learns CORRECT baseload patterns
- ✅ Prediction: Lag features CORRECT, predictions ACCURATE
- ✅ Optimization: Battery simulated on CORRECT predicted load
- ✅ Analysis: Metrics UNBIASED and reliable

---

## Next Steps (If Battery Being Added)

1. **Before Battery Arrives:**
   - Read SENSOR_POWER_ANALYSIS.md Section 11 (Recommendations)
   - Add unit tests from LOAD_CALCULATION_FLOWCHART.md Section "Testing Strategy"
   - Plan sensor extraction for battery_power

2. **When Battery Arrives:**
   - Extract battery_power sensor to Home Assistant
   - Add to ENTITIES in extract_data.py
   - Update process_data.py line 50 formula
   - Update predict_future.py get_baseload_at_lag() function
   - Bump VERSION file (MINOR version - model training changes)

3. **After Battery Integration:**
   - Run train_model.py with --holdout-days to retrain
   - Verify analyze_performance.py shows improved MAE
   - Monitor bias_kw metric for signs of systematic error
   - Compare old predictions vs new predictions on same window

---

## Questions This Answers

1. **How is baseload currently calculated?**
   → `total_home_power = grid_power + solar_actual`
   → `baseload_power = total_home_power - gshp_power - leaf_power`

2. **Where are grid_power, solar_power used?**
   → Extract: extract_data.py
   → Process: process_data.py (to calculate total_home_power)
   → Train: train_model.py (indirectly via baseload)
   → Predict: predict_future.py (to calculate lag features)

3. **Does the model account for battery affecting grid power?**
   → NO - Battery power is not measured or subtracted

4. **Will the system work when battery is installed?**
   → NO - Will systematically overestimate home load

5. **What's the fix?**
   → Extract battery_power sensor
   → Subtract from total_home_power calculation
   → Retrain model on corrected history

---

## Document Statistics

| Document | Lines | Words | Sections |
|----------|-------|-------|----------|
| archive/LOAD-CALC-SEARCH-RESULTS.txt | 200+ | 2,500 | 7 major |
| SENSOR_POWER_ANALYSIS.md | 400+ | 5,000 | 12 major |
| LOAD_CALCULATION_FLOWCHART.md | 300+ | 4,000 | 6 major |
| **Total** | **900+** | **11,500** | **25+** |

**Time to read:** 30-60 minutes (depending on depth)

---

## How to Use These Documents

- **Pin them:** Keep in your IDE/wiki for reference during development
- **Share them:** Attach to PRs when making battery-related changes
- **Update them:** When architecture changes, update the relevant sections
- **Cite them:** Reference specific line numbers when discussing implementation
- **Test against:** Use the formulas and examples to validate changes

---

**Created:** May 30, 2026
**Status:** Complete analysis of current system
**Scope:** Load power calculation and battery impact

For questions about this analysis, refer to the specific document sections listed in the index above.
